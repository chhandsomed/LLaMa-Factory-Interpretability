"""
并发版本的批量测试脚本

主要改进：
1. 使用 ThreadPoolExecutor 实现并发处理
2. 支持自定义并发线程数 (max_workers 参数)
3. 线程安全的结果收集
4. 实时进度显示和中间结果保存
5. 实时保存正确样本：每获得一个正确响应立即保存，避免数据丢失

实时保存特性：
- 每次验证为正确的样本立即追加到正确样本数据集文件
- 使用文件锁确保并发写入安全
- 避免等到所有请求完成后才保存，降低数据丢失风险
- 实时显示正确样本保存进度

使用说明：
- max_workers: 控制并发线程数，建议根据API速率限制调整(默认5)
- 较高的并发数可能触发API速率限制，建议从较小值开始测试
- 每个线程使用独立的OpenAI客户端实例
- 正确样本文件在测试开始时创建，测试过程中实时更新

性能提升：
- 相比串行处理，并发处理可以显著提升测试速度
- 具体提升幅度取决于API响应时间和并发设置
- 实时保存确保即使程序意外中断也不会丢失已验证的正确样本
"""

import os
import json
import re
import threading
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datetime import datetime

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler('train_realtime_save.log', encoding='utf-8')  # 输出到文件
    ]
)


logger = logging.getLogger(__name__)
def load_dataset(file_path):
    """加载测试数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_answer_from_response(response_content):
    """从模型回复中提取答案"""
    # 提取 <answer> 标签中的内容
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_content, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        
        # 提取列表格式的答案，如 ["7,8,9"] 或 []
        list_match = re.search(r'\[(.*?)\]', answer_text)
        if list_match:
            list_content = list_match.group(1).strip()
            if not list_content:  # 空列表 []
                return []
            else:
                # 处理引号包围的内容
                if '"' in list_content or "'" in list_content:
                    # 移除引号并分割
                    cleaned = re.sub(r'["\']', '', list_content)
                    return [cleaned] if cleaned else []
                else:
                    return [list_content] if list_content else []
    
    # 如果没有找到标准格式，尝试直接查找列表
    direct_list_match = re.search(r'\[(.*?)\]', response_content)
    if direct_list_match:
        list_content = direct_list_match.group(1).strip()
        if not list_content:
            return []
        else:
            cleaned = re.sub(r'["\']', '', list_content)
            return [cleaned] if cleaned else []
    
    return None

def compare_answers(predicted, expected):
    """比较预测答案和期望答案"""
    if predicted is None:
        return 0
    
    # 标准化处理
    if isinstance(predicted, list) and len(predicted) == 0:
        predicted = []
    if isinstance(expected, list) and len(expected) == 0:
        expected = []
    
    # 转换为字符串进行比较
    pred_str = str(predicted).strip()
    exp_str = str(expected).strip()
    
    return 1 if pred_str == exp_str else 0

def save_correct_sample_immediately(original_dataset, result, output_file, correct_samples_lock):
    """
    立即保存单个正确样本到数据集文件
    
    Args:
        original_dataset: 原始数据集
        result: 测试结果
        output_file: 输出文件路径
        correct_samples_lock: 文件写入锁
    """
    if result.get('is_correct', 0) != 1:
        return False
    
    sample_id = result['sample_id']
    
    # 获取原始样本
    if sample_id >= len(original_dataset):
        return False
    
    original_sample = original_dataset[sample_id].copy()
    
    # 添加模型的回答内容
    response_content = result.get('response_content', '')
    original_sample['output_content'] = {
        "role": "assistant",
        "content": response_content
    }
    
    # 添加思考过程内容（如果存在）
    reasoning_content = result.get('reasoning_content', '')
    if reasoning_content:
        original_sample['reasoning_content'] = reasoning_content
    
    # 添加测试信息（可选）
    original_sample['test_info'] = {
        "predicted_answer": result.get('predicted_answer'),
        "expected_answer": result.get('expected_answer'),
        "is_correct": result.get('is_correct'),
        "usage": result.get('usage', {}),
        "timestamp": datetime.now().isoformat()
    }
    
    # 线程安全地写入文件
    with correct_samples_lock:
        # 检查文件是否存在，如果不存在则创建空列表
        correct_samples = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    correct_samples = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                correct_samples = []
        
        # 添加新的正确样本
        correct_samples.append(original_sample)
        
        # 保存更新后的数据集
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(correct_samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ 正确样本 {sample_id} 已保存，当前正确样本总数: {len(correct_samples)}")
    return True

def generate_correct_dataset(original_dataset, test_results, start_idx, output_file):
    """
    生成包含正确回答的数据集（批量模式，向后兼容）
    
    Args:
        original_dataset: 原始数据集
        test_results: 测试结果列表
        start_idx: 测试开始索引
        output_file: 输出文件路径
    """
    correct_samples = []
    
    for result in test_results:
        if result.get('is_correct', 0) == 1:
            sample_id = result['sample_id']
            
            # 获取原始样本
            if sample_id < len(original_dataset):
                original_sample = original_dataset[sample_id].copy()
                
                # 添加模型的回答内容
                response_content = result.get('response_content', '')
                original_sample['output_content'] = {
                    "role": "assistant",
                    "content": response_content
                }
                
                # 添加思考过程内容（如果存在）
                reasoning_content = result.get('reasoning_content', '')
                if reasoning_content:
                    original_sample['reasoning_content'] = reasoning_content
                
                # 添加测试信息（可选）
                original_sample['test_info'] = {
                    "predicted_answer": result.get('predicted_answer'),
                    "expected_answer": result.get('expected_answer'),
                    "is_correct": result.get('is_correct'),
                    "usage": result.get('usage', {})
                }
                
                correct_samples.append(original_sample)
    
    # 保存正确样本数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(correct_samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"正确样本数据集已保存至: {output_file}")
    logger.info(f"包含 {len(correct_samples)} 个正确样本")
    
    return correct_samples

def test_single_sample(client, sample, sample_id):
    """测试单个样本"""
    try:
        # 构建消息
        messages = [
            sample["instruction"],
            sample["input"]
        ]
        
        # 调用API
        response = client.chat.completions.create(
            model="ep-20250818191904-hh4v9",
            temperature=0.4,
            timeout=1800,
            messages=messages,
            extra_body={
                "thinking": {
                    "type": "enabled",
                }
            },
        )
        
        # 提取回复内容
        response_content = response.choices[0].message.content
        
        # 提取思考过程内容（reasoning_content）
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '') or ''
        
        # 解析答案
        predicted_answer = extract_answer_from_response(response_content)
        expected_answer = sample["output"]
        
        # 比较答案
        is_correct = compare_answers(predicted_answer, expected_answer)
        
        # 记录结果
        result = {
            "sample_id": sample_id,
            "predicted_answer": predicted_answer,
            "expected_answer": expected_answer,
            "is_correct": is_correct,
            "response_content": response_content,
            "reasoning_content": reasoning_content,
            "usage": {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }
        
        return result
        
    except Exception as e:
        logger.info(f"样本 {sample_id} 处理失败: {str(e)}")
        return {
            "sample_id": sample_id,
            "error": str(e),
            "predicted_answer": None,
            "expected_answer": sample["output"],
            "is_correct": 0
        }

def create_client():
    """创建客户端实例（用于多线程）"""
    return OpenAI(
        api_key="96810a0b-07f3-4191-b067-28d1469b1bd3",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        timeout=1800
    )

def process_sample_wrapper(args):
    """包装函数，用于并发处理"""
    sample, sample_id = args
    # 为每个线程创建独立的客户端
    client = create_client()
    return test_single_sample(client, sample, sample_id)

def batch_test(dataset_path, start_idx=0, end_idx=None, output_dir="results", max_workers=5):
    """并发批量测试"""
    # 加载数据集
    logger.info(f"加载数据集: {dataset_path}")
    dataset = load_dataset(dataset_path)
    
    # 确定测试范围
    if end_idx is None:
        end_idx = len(dataset)
    end_idx = min(end_idx, len(dataset))
    
    logger.info(f"测试范围: {start_idx} - {end_idx}")
    logger.info(f"总样本数: {end_idx - start_idx}")
    logger.info(f"并发线程数: {max_workers}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"test_results_{timestamp}.json")
    
    results = []
    results_lock = threading.Lock()  # 用于线程安全的结果收集
    correct_count = 0
    total_count = 0
    
    # 设置正确样本数据集文件路径和锁
    correct_dataset_file = os.path.join(output_dir, f"correct_samples_with_responses_{timestamp}.json")
    correct_samples_lock = threading.Lock()  # 用于线程安全的正确样本文件写入
    
    # 初始化正确样本文件（创建空列表）
    with open(correct_dataset_file, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    
    logger.info(f"正确样本将实时保存至: {correct_dataset_file}")
    
    # 准备任务列表
    tasks = [(dataset[i], i) for i in range(start_idx, end_idx)]
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_sample = {executor.submit(process_sample_wrapper, task): task[1] for task in tasks}
        
        completed_count = 0
        for future in as_completed(future_to_sample):
            sample_id = future_to_sample[future]
            
            try:
                result = future.result()
                
                # 线程安全地添加结果
                with results_lock:
                    results.append(result)
                    
                    if "is_correct" in result:
                        correct_count += result["is_correct"]
                        total_count += 1
                    
                    completed_count += 1
                
                # 立即保存正确样本（在锁外进行，避免阻塞）
                if result.get('is_correct', 0) == 1:
                    save_correct_sample_immediately(dataset, result, correct_dataset_file, correct_samples_lock)
                
                # 每处理10个样本显示进度（在锁外进行）
                if completed_count % 10 == 0:
                    with results_lock:  # 只在读取统计信息时加锁
                        current_accuracy = correct_count / total_count if total_count > 0 else 0
                        logger.info(f"已完成 {completed_count}/{len(tasks)} 个样本，当前准确率: {current_accuracy:.3f}")
                        
                        # 保存中间结果
                        intermediate_results = {
                            "dataset_path": dataset_path,
                            "test_range": f"{start_idx}-{start_idx + completed_count}",
                            "timestamp": timestamp,
                            "total_samples": completed_count,
                            "correct_samples": correct_count,
                            "accuracy": current_accuracy,
                            "max_workers": max_workers,
                            "results": sorted(results, key=lambda x: x['sample_id']),
                            "correct_dataset_file": correct_dataset_file
                        }
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(intermediate_results, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.info(f"处理样本 {sample_id} 时发生异常: {str(e)}")
                # 添加错误结果
                with results_lock:
                    error_result = {
                        "sample_id": sample_id,
                        "error": str(e),
                        "predicted_answer": None,
                        "expected_answer": dataset[sample_id]["output"] if sample_id < len(dataset) else None,
                        "is_correct": 0
                    }
                    results.append(error_result)
                    total_count += 1
                    completed_count += 1
    
    # 对结果按sample_id排序
    results.sort(key=lambda x: x['sample_id'])
    
    # 计算最终统计信息
    final_results = {
        "dataset_path": dataset_path,
        "test_range": f"{start_idx}-{end_idx}",
        "timestamp": timestamp,
        "total_samples": total_count,
        "correct_samples": correct_count,
        "accuracy": correct_count / total_count if total_count > 0 else 0,
        "max_workers": max_workers,
        "correct_dataset_file": correct_dataset_file,
        "results": results
    }
    
    # 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # 输出统计结果
    logger.info("\n" + "="*50)
    logger.info("测试完成！")
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"测试范围: {start_idx} - {end_idx}")
    logger.info(f"总样本数: {total_count}")
    logger.info(f"正确样本数: {correct_count}")
    logger.info(f"准确率: {correct_count/total_count:.3f} ({correct_count}/{total_count})")
    logger.info(f"结果保存至: {output_file}")
    if correct_count > 0:
        logger.info(f"正确样本数据集: {correct_dataset_file}")
    logger.info("="*50)
    
    return final_results

def analyze_results(results_file):
    """分析测试结果"""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"结果分析 - {results_file}")
    logger.info(f"总样本数: {data['total_samples']}")
    logger.info(f"正确样本数: {data['correct_samples']}")
    logger.info(f"准确率: {data['accuracy']:.3f}")
    
    # 分析错误样本
    error_samples = [r for r in data['results'] if r.get('is_correct', 0) == 0]
    logger.info(f"错误样本数: {len(error_samples)}")
    
    if error_samples:
        logger.info("\n错误样本示例（前5个）:")
        for i, sample in enumerate(error_samples[:5]):
            logger.info(f"样本ID {sample['sample_id']}:")
            logger.info(f"  预测: {sample.get('predicted_answer', 'N/A')}")
            logger.info(f"  期望: {sample.get('expected_answer', 'N/A')}")
            if 'error' in sample:
                logger.info(f"  错误: {sample['error']}")
            logger.info()

if __name__ == "__main__":
    import time
    
    # 使用示例
    # dataset_path = "/home/ch/work/interpretability_research/data_deal/test_dataset_2000.json"
    dataset_path = "/home/ch/work/interpretability_research/data_deal/train_dataset.json"
    
    logger.info("=== 并发批量测试示例（实时保存版本）===")
    logger.info("支持的参数:")
    logger.info("- max_workers: 并发线程数 (建议2-10，根据API限制调整)")
    logger.info("- start_idx, end_idx: 测试数据范围")
    logger.info("- output_dir: 结果输出目录")
    logger.info("\n新增实时保存特性:")
    logger.info("- 每个正确样本验证后立即保存到正确样本数据集")
    logger.info("- 避免等待所有请求完成，降低数据丢失风险")
    logger.info("- 实时显示正确样本保存进度")
    
    # 记录开始时间
    start_time = time.time()
    
    # 测试样本（可以根据需要调整）
    results = batch_test(
        dataset_path=dataset_path,
        output_dir="multi_worker_results_realtime_save_train",
        max_workers=50  # 并发线程数，建议根据API速率限制调整
    )


    # 如果要测试全部样本，可以使用：
    # results = batch_test(
    #     dataset_path=dataset_path,
    #     output_dir="multi_worker_results_realtime_save_test",
    #     max_workers=10  # 可以根据需要调整并发数
    # )
    
    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"\n总耗时: {total_time:.2f} 秒")
    logger.info(f"平均每样本耗时: {total_time/results['total_samples']:.2f} 秒")
    
    # 如果要测试全部样本，可以使用：
    # results = batch_test(
    #     dataset_path=dataset_path,
    #     max_workers=5  # 可以根据需要调整并发数
    # )
    
    # 性能建议
    logger.info("\n=== 性能优化建议 ===")
    logger.info("1. 根据API速率限制调整 max_workers 参数")
    logger.info("2. 如果遇到速率限制错误，请降低并发数")
    logger.info("3. 较大的数据集建议分批处理")
    logger.info("4. 监控API使用量避免超出配额")
    logger.info("5. 实时保存确保即使程序意外中断也不会丢失已验证的正确样本")
