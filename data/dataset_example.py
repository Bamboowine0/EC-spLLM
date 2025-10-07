#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
from dataset_loader import DatasetLoader, load_multiple_datasets, find_datasets_in_directory

def convert_datasets(datasets, output_dir):
    print("Step 1: Converting datasets to unified format...")
    
    cmd = ["python", "dataset_converter.py", "--output_dir", output_dir]
    
    if datasets:
        cmd.extend(["--dataset"] + datasets)
    else:
        cmd.append("--all")
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Dataset conversion completed\n")

def load_and_analyze_dataset(dataset_path):
    print(f"Step 2: Loading dataset {os.path.basename(dataset_path)}...")
    
    loader = DatasetLoader(dataset_path)
    
    print(f"Dataset source: {loader.get_source()}")
    print(f"Dataset size: {loader.get_size()} questions")
    print(f"Dataset type: {loader.get_type()}")
    
    is_mc = "Yes" if loader.is_multiple_choice() else "No"
    has_ctx = "Yes" if loader.has_context() else "No"
    
    print(f"Is multiple choice: {is_mc}")
    print(f"Has context: {has_ctx}")
    
    print("\nData samples:")
    for i in range(min(3, loader.get_size())):
        print(f"\nSample {i+1}:")
        print(f"Question: {loader.get_question(i)}")
        
        if loader.is_multiple_choice():
            choices = loader.get_choices(i)
            choice_str = "\n".join([f"  {idx+1}. {choice}" for idx, choice in enumerate(choices)])
            print(f"Choices:\n{choice_str}")
        
        print(f"Answer: {loader.get_answer(i)}")
        
        if loader.has_context():
            context = loader.get_context(i)
            if len(context) > 100:
                context = context[:100] + "..."
            print(f"Context: {context}")
    
    return loader

def demo_batch_processing(loader):
    print("\nStep 3: Demonstrating batch processing...")
    
    batch_size = 5
    total_size = loader.get_size()
    
    print(f"Using batch size {batch_size} to process {total_size} questions")
    
    for i, batch in enumerate(loader.get_batches(batch_size=batch_size, shuffle=True)):
        if i > 0:
            break
            
        print(f"\nBatch {i+1}:")
        print(f"Batch size: {len(batch)}")
        
        for j, item in enumerate(batch):
            print(f"\n  Item {j+1}:")
            print(f"  Question: {item['question']}")
            
            if 'choices' in item:
                choice_str = "\n".join([f"    {idx+1}. {choice}" for idx, choice in enumerate(item['choices'])])
                print(f"  Choices:\n{choice_str}")
            
            print(f"  Answer: {item['answer']}")
            
            print(f"  [Simulation] Processing question...")

def merge_datasets_demo(dataset_dir):
    print("\nStep 4: Demonstrating merging multiple datasets...")
    
    json_files = find_datasets_in_directory(dataset_dir)
    
    if len(json_files) < 2:
        print(f"Found less than 2 dataset files in {dataset_dir}, skipping merge demo")
        return
    
    files_to_merge = json_files[:2]
    print(f"Will merge the following files:")
    for file in files_to_merge:
        print(f"  - {os.path.basename(file)}")
    
    merged_loader = load_multiple_datasets(files_to_merge)
    
    print("\nMerged dataset information:")
    print(f"Merged from: {merged_loader.get_metadata().get('merged_from', ['Unknown'])}")
    print(f"Merged size: {merged_loader.get_size()} questions")
    
    if merged_loader.get_size() > 0:
        print("\nMerged dataset sample:")
        item = merged_loader.get_item(0)
        print(f"Question: {item['question']}")
        print(f"Answer: {item['answer']}")
        
        if 'choices' in item:
            choice_str = "\n".join([f"  {idx+1}. {choice}" for idx, choice in enumerate(item['choices'])])
            print(f"Choices:\n{choice_str}")

def main():
    parser = argparse.ArgumentParser(description="Dataset tools usage example")
    
    parser.add_argument('--convert', action='store_true', help='Whether to execute data conversion step')
    parser.add_argument('--datasets', nargs='+', choices=['chatdoctor5k', 'pubmed_qa', 'medmc_qa', 'SimpleQA', 'scienceQA', 'sciq'], 
                      help='Datasets to convert, if not specified, convert all supported datasets')
    parser.add_argument('--output_dir', default='./processed_datasets', help='Directory to save processed datasets')
    parser.add_argument('--dataset_file', help='Specific dataset file path to load, if not specified, use the first file in output_dir')
    
    args = parser.parse_args()
    
    if args.convert:
        convert_datasets(args.datasets, args.output_dir)
    
    if args.dataset_file:
        dataset_path = args.dataset_file
    else:
        files = find_datasets_in_directory(args.output_dir)
        if not files:
            print(f"Error: No dataset files found in {args.output_dir}")
            print("Hint: Use --convert parameter to generate dataset files first")
            return
        dataset_path = files[0]
    
    loader = load_and_analyze_dataset(dataset_path)
    
    demo_batch_processing(loader)
    
    merge_datasets_demo(args.output_dir)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 