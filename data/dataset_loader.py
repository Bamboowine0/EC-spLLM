#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一格式数据集加载工具

用于加载由dataset_converter.py处理生成的统一格式JSON文件，
提供简便的API用于访问数据和迭代处理。

用法示例:
    # 加载数据集
    loader = DatasetLoader("./processed_datasets/scienceQA.json")
    
    # 获取数据集信息
    print(f"数据集来源: {loader.get_source()}")
    print(f"数据集大小: {loader.get_size()}")
    print(f"数据集类型: {loader.get_type()}")
    
    # 访问数据
    for i in range(10):  # 处理前10个问题
        question = loader.get_question(i)
        answer = loader.get_answer(i)
        print(f"问题: {question}")
        print(f"答案: {answer}")
        
        # 如果是选择题，获取选项
        if loader.is_multiple_choice():
            choices = loader.get_choices(i)
            print(f"选项: {choices}")
    
    # 使用批处理
    for batch in loader.get_batches(batch_size=32, shuffle=True):
        process_batch(batch)
"""

import os
import json
import random
from typing import List, Dict, Any, Optional, Iterator, Tuple

class DatasetLoader:
    """统一格式数据集加载器"""
    
    def __init__(self, file_path: str):
        """
        初始化数据集加载器
        
        :param file_path: JSON数据文件路径
        """
        self.file_path = file_path
        self._load_data()
        
    def _load_data(self):
        """加载数据文件"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # 进行基本验证
        required_keys = ['questions', 'answers', 'metadata']
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"数据格式错误: 缺少必需的'{key}'字段")
                
        # 检查数据一致性
        n_questions = len(self.data['questions'])
        n_answers = len(self.data['answers'])
        
        if n_questions != n_answers:
            raise ValueError(f"数据不一致: 问题数量({n_questions})与答案数量({n_answers})不匹配")
            
        # 如果有选项，确保选项数量与问题数量一致
        if self.data['choices'] is not None and len(self.data['choices']) != n_questions:
            raise ValueError(f"数据不一致: 问题数量({n_questions})与选项数量({len(self.data['choices'])})不匹配")
            
        # 如果有上下文，确保上下文数量与问题数量一致
        if self.data['contexts'] is not None and len(self.data['contexts']) != n_questions:
            raise ValueError(f"数据不一致: 问题数量({n_questions})与上下文数量({len(self.data['contexts'])})不匹配")
    
    def get_size(self) -> int:
        """获取数据集大小"""
        return self.data['metadata'].get('size', len(self.data['questions']))
    
    def get_source(self) -> str:
        """获取数据集来源"""
        return self.data['metadata'].get('source', 'unknown')
    
    def get_type(self) -> str:
        """获取数据集类型 (multiple_choice 或 short_answer)"""
        return self.data['metadata'].get('type', 'unknown')
    
    def is_multiple_choice(self) -> bool:
        """判断是否为选择题数据集"""
        return self.data['choices'] is not None
    
    def has_context(self) -> bool:
        """判断是否包含上下文信息"""
        return self.data['contexts'] is not None
    
    def get_question(self, index: int) -> str:
        """获取指定索引的问题"""
        self._check_index(index)
        return self.data['questions'][index]
    
    def get_answer(self, index: int) -> str:
        """获取指定索引的答案"""
        self._check_index(index)
        return self.data['answers'][index]
    
    def get_choices(self, index: int) -> Optional[List[str]]:
        """获取指定索引的选项列表"""
        self._check_index(index)
        if not self.is_multiple_choice():
            return None
        return self.data['choices'][index]
    
    def get_context(self, index: int) -> Optional[str]:
        """获取指定索引的上下文"""
        self._check_index(index)
        if not self.has_context():
            return None
        return self.data['contexts'][index]
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取数据集元数据"""
        return self.data['metadata']
    
    def get_all_data(self) -> Dict[str, Any]:
        """获取完整数据"""
        return self.data
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """获取指定索引的完整数据项"""
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
        """
        获取索引列表，可用于迭代访问数据集
        
        :param batch_size: 批量大小，None表示整个数据集
        :param shuffle: 是否打乱顺序
        :return: 索引列表
        """
        total_size = self.get_size()
        indices = list(range(total_size))
        
        if shuffle:
            random.shuffle(indices)
        
        if batch_size is None or batch_size >= total_size:
            return indices
        
        return indices[:batch_size]
    
    def get_batches(self, batch_size: int = 32, shuffle: bool = True) -> Iterator[List[Dict[str, Any]]]:
        """
        获取数据批次迭代器
        
        :param batch_size: 每批数据大小
        :param shuffle: 是否打乱数据顺序
        :return: 批次数据迭代器
        """
        total_size = self.get_size()
        indices = list(range(total_size))
        
        if shuffle:
            random.shuffle(indices)
        
        # 生成批次
        for i in range(0, total_size, batch_size):
            batch_indices = indices[i:min(i + batch_size, total_size)]
            batch = [self.get_item(idx) for idx in batch_indices]
            yield batch
    
    def _check_index(self, index: int):
        """检查索引是否有效"""
        if index < 0 or index >= self.get_size():
            raise IndexError(f"索引越界: {index}，有效范围为 0-{self.get_size()-1}")

def load_multiple_datasets(file_paths: List[str]) -> DatasetLoader:
    """
    加载并合并多个数据集
    
    :param file_paths: 数据集文件路径列表
    :return: 合并后的数据集加载器
    """
    if not file_paths:
        raise ValueError("文件路径列表不能为空")
        
    # 加载第一个数据集作为基础
    base_loader = DatasetLoader(file_paths[0])
    base_data = base_loader.get_all_data()
    
    # 如果只有一个文件，直接返回
    if len(file_paths) == 1:
        return base_loader
    
    # 合并其他数据集
    for file_path in file_paths[1:]:
        loader = DatasetLoader(file_path)
        data = loader.get_all_data()
        
        # 合并问题和答案
        base_data['questions'].extend(data['questions'])
        base_data['answers'].extend(data['answers'])
        
        # 合并选项（如果有）
        if base_data['choices'] is not None and data['choices'] is not None:
            base_data['choices'].extend(data['choices'])
        elif data['choices'] is not None:
            # 如果基础数据没有选项但新数据有，无法合并
            print(f"警告: 数据集 {file_path} 包含选项，但基础数据集没有选项，忽略选项数据")
        
        # 合并上下文（如果有）
        if base_data['contexts'] is not None and data['contexts'] is not None:
            base_data['contexts'].extend(data['contexts'])
        elif data['contexts'] is not None:
            # 如果基础数据没有上下文但新数据有，无法合并
            print(f"警告: 数据集 {file_path} 包含上下文，但基础数据集没有上下文，忽略上下文数据")
            
        # 更新大小信息
        base_data['metadata']['size'] = len(base_data['questions'])
        
        # 添加合并信息
        if 'merged_from' not in base_data['metadata']:
            base_data['metadata']['merged_from'] = [base_data['metadata']['source']]
        
        base_data['metadata']['merged_from'].append(data['metadata']['source'])
    
    # 创建临时文件保存合并结果
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp:
        json.dump(base_data, temp, ensure_ascii=False, indent=2)
        temp_path = temp.name
    
    # 加载合并后的数据集
    merged_loader = DatasetLoader(temp_path)
    
    # 删除临时文件
    os.unlink(temp_path)
    
    return merged_loader

def find_datasets_in_directory(directory: str, pattern: str = "*.json") -> List[str]:
    """
    在指定目录中查找匹配的数据集文件
    
    :param directory: 目录路径
    :param pattern: 文件匹配模式
    :return: 匹配的文件路径列表
    """
    import glob
    pattern_path = os.path.join(directory, pattern)
    return glob.glob(pattern_path)

# 使用示例
if __name__ == "__main__":
    # 查找测试文件
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # 查找默认目录中的第一个JSON文件
        data_dir = "./processed_datasets"
        files = find_datasets_in_directory(data_dir)
        if not files:
            print(f"错误: 在 {data_dir} 中未找到数据集文件")
            print("提示: 先运行 dataset_converter.py 生成数据集文件")
            sys.exit(1)
        test_file = files[0]
    
    try:
        # 加载并测试数据集
        loader = DatasetLoader(test_file)
        print(f"成功加载数据集: {test_file}")
        print(f"数据集来源: {loader.get_source()}")
        print(f"数据集大小: {loader.get_size()}")
        print(f"数据集类型: {loader.get_type()}")
        
        # 打印前3个问题
        print("\n数据样例:")
        for i in range(min(3, loader.get_size())):
            print(f"\n示例 {i+1}:")
            print(f"问题: {loader.get_question(i)}")
            print(f"答案: {loader.get_answer(i)}")
            
            if loader.is_multiple_choice():
                print(f"选项: {loader.get_choices(i)}")
                
            if loader.has_context():
                context = loader.get_context(i)
                # 如果上下文很长，只显示前100个字符
                if context and len(context) > 100:
                    context = context[:100] + "..."
                print(f"上下文: {context}")
                
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1) 