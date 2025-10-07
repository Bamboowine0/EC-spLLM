#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据集格式转换工具

将不同格式的数据集统一转换为标准JSON格式，便于后续使用。
支持命令行参数配置，可以批量处理多个数据集。

用法示例:
    # 处理单个数据集
    python dataset_converter.py --dataset scienceQA --output_dir ./processed_datasets
    
    # 处理多个数据集
    python dataset_converter.py --dataset scienceQA pubmed_qa --output_dir ./processed_datasets
    
    # 处理所有支持的数据集
    python dataset_converter.py --all --output_dir ./processed_datasets
"""

import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

def prepare_dataset(dataset, subject="natural science"):
    """
    将不同格式的数据集统一处理成标准格式
    
    返回格式统一为：
    {
        'questions': list, # 问题列表
        'answers': list,   # 答案列表
        'choices': list,   # 选项列表（如果有），否则为None
        'contexts': list,  # 上下文列表（如果有），否则为None
        'metadata': dict   # 其他元数据
    }
    
    :param dataset: 数据集名称
    :param subject: 当dataset为scienceQA时使用的学科类型
    :return: 统一格式的数据字典
    """
    result = {
        'questions': [],
        'answers': [],
        'choices': None,
        'contexts': None,
        'metadata': {}
    }
    
    # 医疗类
    if dataset == "chatdoctor5k":
        file = "./original_datasets/" + str(dataset) + ".json"
        with open(file, encoding='utf-8') as f:
            datas = json.load(f)
        
        result['questions'] = [item['input'] for item in datas]
        result['answers'] = [item['output'] for item in datas]
        result['metadata']['source'] = 'chatdoctor5k'
        result['metadata']['type'] = 'short_answer'
        
    elif dataset == "pubmed_qa":  # 转换成选择题，有"yes"、"maybe"、"no"三个选项
        datas = load_dataset("pubmed_qa", 'pqa_labeled')['train']
        
        result['questions'] = datas['question']
        result['answers'] = datas['final_decision']
        result['choices'] = [["yes", "no", "maybe"] for _ in range(len(datas['question']))]
        result['contexts'] = datas['context']
        result['metadata']['long_answers'] = datas['long_answer']
        result['metadata']['source'] = 'pubmed_qa'
        result['metadata']['type'] = 'multiple_choice'
        
    elif dataset == "medmc_qa":
        datas = load_dataset('openlifescienceai/medmcqa')['validation'].filter(
            lambda example: example['choice_type'] == "single")
        
        result['questions'] = datas['question']
        result['answers'] = datas['cop']
        
        choices = []
        for i in range(len(datas["question"])):
            choices.append([datas["opa"][i], datas["opb"][i], datas["opc"][i], datas["opd"][i]])
        result['choices'] = choices
        result['metadata']['source'] = 'medmc_qa'
        result['metadata']['type'] = 'multiple_choice'
        
    # 通识类
    elif dataset == "SimpleQA":
        file = "./original_datasets/" + str(dataset) + ".json"
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result['questions'] = [item['question'] for item in data]
        result['answers'] = [item['answer'] for item in data]
        result['metadata']['source'] = 'SimpleQA'
        result['metadata']['type'] = 'short_answer'
        
    elif dataset == "scienceQA":
        d = load_dataset("derek-thomas/ScienceQA")
        datas = d['train'].filter(
            lambda example: example['subject'] == subject and example['image'] is None and len(example['hint']) == 0)
        
        result['questions'] = datas['question']
        result['answers'] = datas['answer']
        result['choices'] = datas['choices']
        result['metadata']['source'] = 'scienceQA'
        result['metadata']['subject'] = subject
        result['metadata']['type'] = 'multiple_choice'
        
    elif dataset == "Chinese_SimpleQA":
        ds = load_dataset("OpenStellarTeam/Chinese-SimpleQA")
        datas = ds['train']
        result['questions'] = datas['question']
        result['answers'] = datas['answer']
        result['metadata']['source'] = 'Chinese_SimpleQA'
        result['metadata']['type'] = 'short_answer'
        
    else:  # 默认为sciq
        datas = load_dataset('allenai/sciq')['test']
        
        result['questions'] = datas['question']
        result['answers'] = datas['correct_answer']
        result['metadata']['source'] = 'sciq'
        result['metadata']['type'] = 'short_answer'
    
        
    # 添加数据集大小信息
    result['metadata']['size'] = len(result['questions'])
        
    return result

def save_dataset_to_json(dataset_result, output_dir, dataset_name, split=False, max_items_per_file=10000):
    """
    将处理后的数据集保存为JSON文件
    
    :param dataset_result: prepare_dataset返回的结果
    :param output_dir: 输出目录
    :param dataset_name: 数据集名称
    :param split: 是否将大数据集拆分为多个文件
    :param max_items_per_file: 每个文件最大的条目数
    :return: 保存的文件路径列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    total_size = dataset_result['metadata']['size']
    saved_files = []
    
    # 如果数据集较小或不需要拆分，直接保存为一个文件
    if not split or total_size <= max_items_per_file:
        output_file = os.path.join(output_dir, f"{dataset_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_result, f, ensure_ascii=False, indent=2)
        saved_files.append(output_file)
        print(f"数据集 {dataset_name} 已保存到 {output_file}，共 {total_size} 条数据")
        return saved_files
    
    # 需要拆分大数据集
    num_files = (total_size + max_items_per_file - 1) // max_items_per_file  # 向上取整
    
    for i in range(num_files):
        start_idx = i * max_items_per_file
        end_idx = min((i + 1) * max_items_per_file, total_size)
        
        # 创建当前分片的数据子集
        subset = {
            'questions': dataset_result['questions'][start_idx:end_idx],
            'answers': dataset_result['answers'][start_idx:end_idx],
            'metadata': dataset_result['metadata'].copy()
        }
        
        # 处理可能为None的字段
        if dataset_result['choices'] is not None:
            subset['choices'] = dataset_result['choices'][start_idx:end_idx]
        else:
            subset['choices'] = None
            
        if dataset_result['contexts'] is not None:
            subset['contexts'] = dataset_result['contexts'][start_idx:end_idx]
        else:
            subset['contexts'] = None
        
        # 更新元数据
        subset['metadata']['size'] = end_idx - start_idx
        subset['metadata']['split_info'] = f"第 {i+1}/{num_files} 分片"
        
        # 保存分片
        output_file = os.path.join(output_dir, f"{dataset_name}_part{i+1}of{num_files}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        saved_files.append(output_file)
        
        print(f"数据集 {dataset_name} 分片 {i+1}/{num_files} 已保存到 {output_file}，包含 {end_idx - start_idx} 条数据")
    
    return saved_files

def get_supported_datasets():
    """
    返回当前支持的所有数据集名称
    """
    return ["chatdoctor5k", "pubmed_qa", "medmc_qa", "SimpleQA", "scienceQA", "sciq", "Chinese_SimpleQA"]

def main():
    parser = argparse.ArgumentParser(description="将不同格式的数据集统一转换为标准JSON格式")
    
    # 数据集参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', nargs='+', choices=get_supported_datasets(), 
                      help='要处理的数据集名称，可指定多个')
    group.add_argument('--all', action='store_true', help='处理所有支持的数据集')
    
    # 输出参数
    parser.add_argument('--output_dir', default='./processed_datasets', help='处理后的数据集保存目录')
    parser.add_argument('--split', action='store_true', help='是否将大数据集拆分为多个文件')
    parser.add_argument('--max_items_per_file', type=int, default=10000, help='拆分时每个文件最大的条目数')
    
    # scienceQA特定参数
    parser.add_argument('--subject', default='natural science', 
                     choices=['natural science', 'language science', 'social science'], 
                     help='scienceQA数据集的学科类型')
    
    args = parser.parse_args()
    
    # 确定要处理的数据集
    if args.all:
        datasets_to_process = get_supported_datasets()
    else:
        datasets_to_process = args.dataset
    
    # 处理每个数据集
    for dataset_name in datasets_to_process:
        print(f"正在处理数据集: {dataset_name}...")
        
        # 特殊处理scienceQA数据集
        if dataset_name == "scienceQA":
            dataset_result = prepare_dataset(dataset_name, subject=args.subject)
        else:
            dataset_result = prepare_dataset(dataset_name)
        
        # 保存处理后的数据集
        saved_files = save_dataset_to_json(
            dataset_result, 
            args.output_dir, 
            dataset_name, 
            split=args.split, 
            max_items_per_file=args.max_items_per_file
        )
        
        print(f"数据集 {dataset_name} 处理完成。")
    
    print(f"所有数据集处理完成，结果保存在 {args.output_dir} 目录下。")

if __name__ == "__main__":
    main() 