import json
import random
import re

import numpy as np
from datasets import load_dataset


def remove_leading_comments(java_code):
    """
    去除Java代码中开头的license注释
    :param java_code: Java代码字符串
    :return: 去除注释后的Java代码字符串
    """
    # 去除开头的 /**/ 多行注释
    java_code = re.sub(r'^/\*[\s\S]*?\*/', '', java_code, flags=re.MULTILINE)
    # 去除开头的多个 // 单行注释
    java_code = re.sub(r'^//.*', '', java_code, flags=re.MULTILINE)
    # 去除多余的空行
    java_code = re.sub(r'^\s*$', '', java_code, flags=re.MULTILINE)
    return java_code.lstrip()


def count_non_empty_lines(code_str):
    """
    计算代码字符串中非空行的数量
    :param code_str: 代码字符串
    :return: 非空行的数量
    """
    lines = code_str.strip().splitlines()
    return sum(1 for line in lines if line.strip())


def generate_prompt_response(code: str, max_hole_lines: int = 6, max_hole_ratio: float = 0.2):
    """
    生成prompt和response
    :param code: 代码字符串
    :param max_hole_lines: 最大挖取的行数
    :param max_hole_ratio: 最大挖取的行数占比
    :return: prompt和response
    """
    bos = "<｜fim▁begin｜>"
    eos = "<｜fim▁end｜>"
    hole = "<｜fim▁hole｜>"

    lines = code.splitlines()
    num_lines = np.random.randint(1, min(max_hole_lines, int(len(lines)*max_hole_ratio)) + 1)  # 随机选择挖取的行数
    start_line = np.random.randint(0, len(lines) - num_lines)  # 随机选择起始行

    # 构造response
    response = '\n'.join(lines[start_line:start_line + num_lines])
    if start_line + num_lines < len(lines):
        response += '\n'

    # 构造prompt
    prompt = '\n'.join(lines[:start_line]) + '\n' + hole + '\n'.join(lines[start_line + num_lines:])
    prompt = bos + prompt + eos

    return prompt, response


if __name__ == "__main__":
    # 参数设置
    data_files = "/home/cti/project/python/data/datasets--hrishizone--Java-GitHub-Codes/*.parquet"
    train_save_path = "./data/java_fim.json"
    eval_save_path = "./data/java_fim_eval.json"
    num_samples = 100000  # 采样数量
    num_samples_per_code = 5  # 每个代码文件采样数量
    train_eval_split = 0.99  # 训练集和验证集划分比例
    max_length = 8192  # 上下文最大长度
    min_lines = 10  # 代码最少行数
    max_hole_lines = 6  # 最大挖取行数

    # 加载数据集
    dataset = load_dataset('parquet', data_files=data_files, streaming=True)["train"]
    dataset = dataset.shuffle()

    # 生成json数据集
    json_file = []
    for row in dataset:
        # 特殊符号处理、去除license注释
        code = row['code'].replace('\r', '').expandtabs(4)
        code = remove_leading_comments(code)
        num_lines = count_non_empty_lines(code)

        if len(code) > max_length or num_lines < min_lines:  # 代码不符合要求
            continue

        for _ in range(num_samples_per_code):
            # 生成prompt和response
            prompt, response = '', ''
            while not response or response.isspace():  # 检查response是否为空或仅包含空格，换行符
                prompt, response = generate_prompt_response(code, max_hole_lines)

            # 添加到json文件
            json_obj = {
                "prompt": prompt,
                "response": response,
            }
            json_file.append(json_obj)

        # 控制采样数量
        num_samples -= num_samples_per_code
        if num_samples <= 0:
            break

    # 划分训练集和验证集
    random.shuffle(json_file)
    split_index = int(len(json_file) * train_eval_split)
    train_data = json_file[:split_index]
    eval_data = json_file[split_index:]

    # 保存json数据集
    with open(train_save_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(eval_save_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
