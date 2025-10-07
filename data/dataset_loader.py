#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
from typing import List, Dict, Any, Optional, Iterator, Tuple

class DatasetLoader:
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._load_data()
        
    def _load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        required_keys = ['questions', 'answers', 'metadata']
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Data format error: missing required '{key}' field")
                
        n_questions = len(self.data['questions'])
        n_answers = len(self.data['answers'])
        
        if n_questions != n_answers:
            raise ValueError(f"Data inconsistency: question count({n_questions}) does not match answer count({n_answers})")
            
        if self.data['choices'] is not None and len(self.data['choices']) != n_questions:
            raise ValueError(f"Data inconsistency: question count({n_questions}) does not match choice count({len(self.data['choices'])})")
            
        if self.data['contexts'] is not None and len(self.data['contexts']) != n_questions:
            raise ValueError(f"Data inconsistency: question count({n_questions}) does not match context count({len(self.data['contexts'])})")
    
    def get_size(self) -> int:
        return self.data['metadata'].get('size', len(self.data['questions']))
    
    def get_source(self) -> str:
        return self.data['metadata'].get('source', 'unknown')
    
    def get_type(self) -> str:
        return self.data['metadata'].get('type', 'unknown')
    
    def is_multiple_choice(self) -> bool:
        return self.data['choices'] is not None
    
    def has_context(self) -> bool:
        return self.data['contexts'] is not None
    
    def get_question(self, index: int) -> str:
        self._check_index(index)
        return self.data['questions'][index]
    
    def get_answer(self, index: int) -> str:
        self._check_index(index)
        return self.data['answers'][index]
    
    def get_choices(self, index: int) -> Optional[List[str]]:
        self._check_index(index)
        if not self.is_multiple_choice():
            return None
        return self.data['choices'][index]
    
    def get_context(self, index: int) -> Optional[str]:
        self._check_index(index)
        if not self.has_context():
            return None
        return self.data['contexts'][index]
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.data['metadata']
    
    def get_all_data(self) -> Dict[str, Any]:
        return self.data
    
    def get_item(self, index: int) -> Dict[str, Any]:
        self._check_index(index)
        item = {
            'question': self.get_question(index),
            'answer': self.get_answer(index)
        }
        
        if self.is_multiple_choice():
            item['choices'] = self.get_choices(index)
            
        if self.has_context():
            item['context'] = self.get_context(index)
            
        return item
    
    def get_indices(self, batch_size: Optional[int] = None, shuffle: bool = False) -> List[int]:
        total_size = self.get_size()
        indices = list(range(total_size))
        
        if shuffle:
            random.shuffle(indices)
        
        if batch_size is None or batch_size >= total_size:
            return indices
        
        return indices[:batch_size]
    
    def get_batches(self, batch_size: int = 32, shuffle: bool = True) -> Iterator[List[Dict[str, Any]]]:
        total_size = self.get_size()
        indices = list(range(total_size))
        
        if shuffle:
            random.shuffle(indices)
        
        for i in range(0, total_size, batch_size):
            batch_indices = indices[i:min(i + batch_size, total_size)]
            batch = [self.get_item(idx) for idx in batch_indices]
            yield batch
    
    def _check_index(self, index: int):
        if index < 0 or index >= self.get_size():
            raise IndexError(f"Index out of range: {index}, valid range is 0-{self.get_size()-1}")

def load_multiple_datasets(file_paths: List[str]) -> DatasetLoader:
    if not file_paths:
        raise ValueError("File path list cannot be empty")
        
    base_loader = DatasetLoader(file_paths[0])
    base_data = base_loader.get_all_data()
    
    if len(file_paths) == 1:
        return base_loader
    
    for file_path in file_paths[1:]:
        loader = DatasetLoader(file_path)
        data = loader.get_all_data()
        
        base_data['questions'].extend(data['questions'])
        base_data['answers'].extend(data['answers'])
        
        if base_data['choices'] is not None and data['choices'] is not None:
            base_data['choices'].extend(data['choices'])
        elif data['choices'] is not None:
            # 如果基础数据没有选项但新数据有，无法合并
            print(f"Warning: Dataset {file_path} contains choices, but base dataset has no choices, ignoring choice data")
        
        if base_data['contexts'] is not None and data['contexts'] is not None:
            base_data['contexts'].extend(data['contexts'])
        elif data['contexts'] is not None:
            # 如果基础数据没有上下文但新数据有，无法合并
            print(f"Warning: Dataset {file_path} contains context, but base dataset has no context, ignoring context data")
            
        base_data['metadata']['size'] = len(base_data['questions'])
        
        if 'merged_from' not in base_data['metadata']:
            base_data['metadata']['merged_from'] = [base_data['metadata']['source']]
        
        base_data['metadata']['merged_from'].append(data['metadata']['source'])
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp:
        json.dump(base_data, temp, ensure_ascii=False, indent=2)
        temp_path = temp.name
    
    merged_loader = DatasetLoader(temp_path)
    
    os.unlink(temp_path)
    
    return merged_loader

def find_datasets_in_directory(directory: str, pattern: str = "*.json") -> List[str]:
    import glob
    pattern_path = os.path.join(directory, pattern)
    return glob.glob(pattern_path)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        data_dir = "./processed_datasets"
        files = find_datasets_in_directory(data_dir)
        if not files:
            print(f"Error: No dataset files found in {data_dir}")
            print("Hint: Run dataset_converter.py first to generate dataset files")
            sys.exit(1)
        test_file = files[0]
    
    try:
        loader = DatasetLoader(test_file)
        print(f"Successfully loaded dataset: {test_file}")
        print(f"Dataset source: {loader.get_source()}")
        print(f"Dataset size: {loader.get_size()}")
        print(f"Dataset type: {loader.get_type()}")
        
        print("\nData samples:")
        for i in range(min(3, loader.get_size())):
            print(f"\nSample {i+1}:")
            print(f"Question: {loader.get_question(i)}")
            print(f"Answer: {loader.get_answer(i)}")
            
            if loader.is_multiple_choice():
                print(f"Choices: {loader.get_choices(i)}")
                
            if loader.has_context():
                context = loader.get_context(i)
                if context and len(context) > 100:
                    context = context[:100] + "..."
                print(f"Context: {context}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 