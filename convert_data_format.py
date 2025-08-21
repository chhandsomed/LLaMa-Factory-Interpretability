#!/usr/bin/env python3
"""
将训练数据从字典格式转换为字符串格式
将 instruction, input, output 从字典形式的 {"role": "xxx", "content": "xxx"} 
转换为直接的字符串形式 (取content的内容)
"""

import json
import os
from pathlib import Path

def convert_sample(sample):
    """转换单个样本的格式"""
    converted = {}
    
    # 转换 instruction
    if isinstance(sample.get('instruction'), dict) and 'content' in sample['instruction']:
        converted['instruction'] = sample['instruction']['content']
    else:
        converted['instruction'] = sample.get('instruction', '')
    
    # 转换 input  
    if isinstance(sample.get('input'), dict) and 'content' in sample['input']:
        converted['input'] = sample['input']['content']
    else:
        converted['input'] = sample.get('input', '')
    
    # 转换 output (处理可能的字典格式)
    if isinstance(sample.get('output'), dict) and 'content' in sample['output']:
        converted['output'] = sample['output']['content']
    else:
        converted['output'] = sample.get('output', '')
    
    # 保留其他字段
    for key, value in sample.items():
        if key not in ['instruction', 'input', 'output']:
            converted[key] = value
    
    return converted

def convert_file(input_path, output_path):
    """转换整个文件"""
    print(f"处理文件: {input_path}")
    
    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换数据
    converted_data = []
    for i, sample in enumerate(data):
        try:
            converted_sample = convert_sample(sample)
            converted_data.append(converted_sample)
        except Exception as e:
            print(f"警告: 处理第 {i+1} 个样本时出错: {e}")
            continue
    
    # 写入转换后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成: {input_path} -> {output_path}")
    print(f"原始样本数: {len(data)}, 转换后样本数: {len(converted_data)}")

def main():
    # 文件路径
    base_dir = Path("/home/ch/work/interpretability_research/lamafactory/LLaMa-Factory-Interpretability/data/test_sft/rawdata")
    
    files_to_convert = [
        "correct_samples_with_responses_20250818_train.json",
        "correct_samples_with_responses_20250818_test.json"
    ]
    
    for filename in files_to_convert:
        input_path = base_dir / filename
        output_path = base_dir / filename  # 直接覆盖原文件
        
        if input_path.exists():
            # 先备份原文件
            backup_path = base_dir / f"{filename}.backup"
            print(f"备份原文件: {input_path} -> {backup_path}")
            with open(input_path, 'r', encoding='utf-8') as src, open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            
            # 转换文件
            convert_file(input_path, output_path)
        else:
            print(f"文件不存在: {input_path}")

if __name__ == "__main__":
    main()
