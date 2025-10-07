#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据集工具使用示例

这个脚本展示了如何使用 dataset_converter.py 和 dataset_loader.py 工具
来处理、加载和使用不同数据集。
"""

import os
import argparse
import subprocess
from dataset_loader import DatasetLoader, load_multiple_datasets, find_datasets_in_directory

def convert_datasets(datasets, output_dir):
    """转换数据集为统一格式"""
    print("第1步：转换数据集为统一格式...")
    
    # 构建命令
    cmd = ["python", "dataset_converter.py", "--output_dir", output_dir]
    
    if datasets:
        cmd.extend(["--dataset"] + datasets)
    else:
        cmd.append("--all")
    
    # 执行命令
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("数据集转换完成\n")

def load_and_analyze_dataset(dataset_path):
    """加载并分析单个数据集"""
    print(f"第2步：加载数据集 {os.path.basename(dataset_path)}...")
    
    loader = DatasetLoader(dataset_path)
    
    # 打印数据集基本信息
    print(f"数据集来源: {loader.get_source()}")
    print(f"数据集大小: {loader.get_size()} 个问题")
    print(f"数据集类型: {loader.get_type()}")
    
    is_mc = "是" if loader.is_multiple_choice() else "否"
    has_ctx = "是" if loader.has_context() else "否"
    
    print(f"是否为选择题: {is_mc}")
    print(f"是否包含上下文: {has_ctx}")
    
    # 显示数据样例
    print("\n数据样例:")
    for i in range(min(3, loader.get_size())):
        print(f"\n示例 {i+1}:")
        print(f"问题: {loader.get_question(i)}")
        
        if loader.is_multiple_choice():
            choices = loader.get_choices(i)
            choice_str = "\n".join([f"  {idx+1}. {choice}" for idx, choice in enumerate(choices)])
            print(f"选项:\n{choice_str}")
        
        print(f"答案: {loader.get_answer(i)}")
        
        if loader.has_context():
            context = loader.get_context(i)
            # 如果上下文很长，只显示前100个字符
            if len(context) > 100:
                context = context[:100] + "..."
            print(f"上下文: {context}")
    
    return loader

def demo_batch_processing(loader):
    """演示批处理数据集"""
    print("\n第3步：演示批处理数据集...")
    
    batch_size = 5
    total_size = loader.get_size()
    
    print(f"使用批大小 {batch_size} 处理 {total_size} 个问题")
    
    # 只显示第一个批次
    for i, batch in enumerate(loader.get_batches(batch_size=batch_size, shuffle=True)):
        if i > 0:
            break
            
        print(f"\n批次 {i+1}:")
        print(f"批次大小: {len(batch)}")
        
        # 模拟处理批次的每个项目
        for j, item in enumerate(batch):
            print(f"\n  项目 {j+1}:")
            print(f"  问题: {item['question']}")
            
            if 'choices' in item:
                choice_str = "\n".join([f"    {idx+1}. {choice}" for idx, choice in enumerate(item['choices'])])
                print(f"  选项:\n{choice_str}")
            
            print(f"  答案: {item['answer']}")
            
            # 模拟处理
            print(f"  [模拟处理] 正在处理问题...")

def merge_datasets_demo(dataset_dir):
    """演示合并多个数据集"""
    print("\n第4步：演示合并多个数据集...")
    
    # 查找所有JSON文件
    json_files = find_datasets_in_directory(dataset_dir)
    
    if len(json_files) < 2:
        print(f"在 {dataset_dir} 中找到的数据集文件少于2个，跳过合并演示")
        return
    
    # 选择前两个文件进行合并
    files_to_merge = json_files[:2]
    print(f"将合并以下文件:")
    for file in files_to_merge:
        print(f"  - {os.path.basename(file)}")
    
    # 合并数据集
    merged_loader = load_multiple_datasets(files_to_merge)
    
    # 显示合并后的信息
    print("\n合并后的数据集信息:")
    print(f"合并来源: {merged_loader.get_metadata().get('merged_from', ['未知'])}")
    print(f"合并大小: {merged_loader.get_size()} 个问题")
    
    # 显示一个样例
    if merged_loader.get_size() > 0:
        print("\n合并数据集样例:")
        item = merged_loader.get_item(0)
        print(f"问题: {item['question']}")
        print(f"答案: {item['answer']}")
        
        if 'choices' in item:
            choice_str = "\n".join([f"  {idx+1}. {choice}" for idx, choice in enumerate(item['choices'])])
            print(f"选项:\n{choice_str}")

def main():
    parser = argparse.ArgumentParser(description="数据集工具使用示例")
    
    parser.add_argument('--convert', action='store_true', help='是否执行数据转换步骤')
    parser.add_argument('--datasets', nargs='+', choices=['chatdoctor5k', 'pubmed_qa', 'medmc_qa', 'SimpleQA', 'scienceQA', 'sciq'], 
                      help='要转换的数据集，不指定则转换所有支持的数据集')
    parser.add_argument('--output_dir', default='./processed_datasets', help='处理后的数据集保存目录')
    parser.add_argument('--dataset_file', help='要加载的特定数据集文件路径，不指定则使用output_dir中的第一个文件')
    
    args = parser.parse_args()
    
    # 第1步：转换数据集（如果需要）
    if args.convert:
        convert_datasets(args.datasets, args.output_dir)
    
    # 查找数据集文件
    if args.dataset_file:
        dataset_path = args.dataset_file
    else:
        files = find_datasets_in_directory(args.output_dir)
        if not files:
            print(f"错误: 在 {args.output_dir} 中未找到数据集文件")
            print("提示: 使用 --convert 参数先生成数据集文件")
            return
        dataset_path = files[0]
    
    # 第2步：加载并分析数据集
    loader = load_and_analyze_dataset(dataset_path)
    
    # 第3步：演示批处理
    demo_batch_processing(loader)
    
    # 第4步：演示合并数据集
    merge_datasets_demo(args.output_dir)
    
    print("\n演示完成!")

if __name__ == "__main__":
    main() 