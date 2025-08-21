import json
import os

def convert_json_file(input_path, output_path):
    """
    将JSON文件中的instruction、input、output从字典转换为content字符串
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    for item in data:
        converted_item = {}
        # 处理instruction字段
        if 'instruction' in item and isinstance(item['instruction'], dict):
            converted_item['instruction'] = item['instruction'].get('content', '')
        # 处理input字段
        if 'input' in item and isinstance(item['input'], dict):
            converted_item['input'] = item['input'].get('content', '')
        # 处理output字段
        if 'output' in item and isinstance(item['output'], dict):
            converted_item['output'] = item['output'].get('content', '')
        # 保留其他字段（如output_origin等）
        for key in item:
            if key not in ['instruction', 'input', 'output']:
                converted_item[key] = item[key]
        converted_data.append(converted_item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

# 处理训练集和测试集文件
train_input = './correct_samples_with_responses_20250818_train.json'
train_output = './converted_train.json'
test_input = './correct_samples_with_responses_20250818_test.json'
test_output = './converted_test.json'

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(train_output), exist_ok=True)

# 执行转换
convert_json_file(train_input, train_output)
convert_json_file(test_input, test_output)

print("转换完成！")