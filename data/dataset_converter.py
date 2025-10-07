#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

def prepare_dataset(dataset, subject="natural science"):
    result = {
        'questions': [],
        'answers': [],
        'choices': None,
        'contexts': None,
        'metadata': {}
    }
    
    if dataset == "chatdoctor5k":
        file = "./original_datasets/" + str(dataset) + ".json"
        with open(file, encoding='utf-8') as f:
            datas = json.load(f)
        
        result['questions'] = [item['input'] for item in datas]
        result['answers'] = [item['output'] for item in datas]
        result['metadata']['source'] = 'chatdoctor5k'
        result['metadata']['type'] = 'short_answer'
        
    elif dataset == "pubmed_qa":
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
        
    else:
        datas = load_dataset('allenai/sciq')['test']
        
        result['questions'] = datas['question']
        result['answers'] = datas['correct_answer']
        result['metadata']['source'] = 'sciq'
        result['metadata']['type'] = 'short_answer'
    
    result['metadata']['size'] = len(result['questions'])
        
    return result

def save_dataset_to_json(dataset_result, output_dir, dataset_name, split=False, max_items_per_file=10000):
    os.makedirs(output_dir, exist_ok=True)
    
    total_size = dataset_result['metadata']['size']
    saved_files = []
    
    if not split or total_size <= max_items_per_file:
        output_file = os.path.join(output_dir, f"{dataset_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_result, f, ensure_ascii=False, indent=2)
        saved_files.append(output_file)
        print(f"Dataset {dataset_name} saved to {output_file}, total {total_size} items")
        return saved_files
    
    num_files = (total_size + max_items_per_file - 1) // max_items_per_file
    
    for i in range(num_files):
        start_idx = i * max_items_per_file
        end_idx = min((i + 1) * max_items_per_file, total_size)
        
        subset = {
            'questions': dataset_result['questions'][start_idx:end_idx],
            'answers': dataset_result['answers'][start_idx:end_idx],
            'metadata': dataset_result['metadata'].copy()
        }
        
        if dataset_result['choices'] is not None:
            subset['choices'] = dataset_result['choices'][start_idx:end_idx]
        else:
            subset['choices'] = None
            
        if dataset_result['contexts'] is not None:
            subset['contexts'] = dataset_result['contexts'][start_idx:end_idx]
        else:
            subset['contexts'] = None
        
        subset['metadata']['size'] = end_idx - start_idx
        subset['metadata']['split_info'] = f"Part {i+1}/{num_files}"
        
        output_file = os.path.join(output_dir, f"{dataset_name}_part{i+1}of{num_files}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        saved_files.append(output_file)
        
        print(f"Dataset {dataset_name} part {i+1}/{num_files} saved to {output_file}, contains {end_idx - start_idx} items")
    
    return saved_files

def get_supported_datasets():
    return ["chatdoctor5k", "pubmed_qa", "medmc_qa", "SimpleQA", "scienceQA", "sciq", "Chinese_SimpleQA"]

def main():
    parser = argparse.ArgumentParser(description="Convert datasets of different formats to standard JSON format")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', nargs='+', choices=get_supported_datasets(), 
                      help='Dataset names to process, can specify multiple')
    group.add_argument('--all', action='store_true', help='Process all supported datasets')
    
    parser.add_argument('--output_dir', default='./processed_datasets', help='Directory to save processed datasets')
    parser.add_argument('--split', action='store_true', help='Whether to split large datasets into multiple files')
    parser.add_argument('--max_items_per_file', type=int, default=10000, help='Maximum items per file when splitting')
    
    parser.add_argument('--subject', default='natural science', 
                     choices=['natural science', 'language science', 'social science'], 
                     help='Subject type for scienceQA dataset')
    
    args = parser.parse_args()
    
    if args.all:
        datasets_to_process = get_supported_datasets()
    else:
        datasets_to_process = args.dataset
    
    for dataset_name in datasets_to_process:
        print(f"Processing dataset: {dataset_name}...")
        
        if dataset_name == "scienceQA":
            dataset_result = prepare_dataset(dataset_name, subject=args.subject)
        else:
            dataset_result = prepare_dataset(dataset_name)
        
        saved_files = save_dataset_to_json(
            dataset_result, 
            args.output_dir, 
            dataset_name, 
            split=args.split, 
            max_items_per_file=args.max_items_per_file
        )
        
        print(f"Dataset {dataset_name} processing completed.")
    
    print(f"All datasets processing completed, results saved in {args.output_dir} directory.")

if __name__ == "__main__":
    main() 