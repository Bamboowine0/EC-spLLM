#!/usr/bin/env python
# -*- coding: utf-8 -*-

opeani_api_keys = {
    "your_key": "sk-xxxx"
}


import os
import argparse
import time
import logging
from datetime import datetime
from pipeline import EC_spLLM_pip
from data.dataset_loader import DatasetLoader
from db_utils import init_db
from var import *
import random
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():

    parser = argparse.ArgumentParser(description="EC-spLLM run script")
    
    # Dataset parameters
    parser.add_argument('--dataset', default='medmc_qa',
                      help='Path to the unified format dataset JSON file')
    parser.add_argument('--mode', default="rephrase",
                      help='Dataset type (rephrase / ori)')
    
    # Test parameters
    parser.add_argument('--test_num', type=int, default=400,
                      help='Number of tests')
    parser.add_argument('--random_seed', type=int, default=23,
                      help='Random seed')
    
    # Knowledge graph parameters
    parser.add_argument('--db_path', default='./chroma',
                      help='ChromaDB database path')
    now_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    kg_name=now_time+str(random.randint(1,1000000))
    parser.add_argument('--kg_name', default=kg_name,
                      help='Knowledge graph name')
    parser.add_argument('--dis_function', default='cosine',
                      help='Knowledge graph distance function type (l2, ip, cosine)')
    
    # LLM parameters
    parser.add_argument('--model', default='gpt-4.1',
                      help='LLM to use')
    parser.add_argument('--openai_api_keys', default=opeani_api_keys["your_key"],
                      help='OpenAI API key')
    
    # Retrieval parameters
    parser.add_argument('--depth', type=int, default=3,
                      help='Maximum retrieval depth')
    parser.add_argument('--width', type=int, default=5,
                      help='Retrieval width (maximum number of entities per layer)')
    parser.add_argument('--dist', type=float, default=0.40,
                      help='Retrieval distance threshold')
    parser.add_argument('--sim_threshold', type=float, default=0.40,
                      help='Similarity threshold')
    parser.add_argument('--sim_top', type=int, default=3,
                      help='Top N after similarity ranking')
    
    # Result saving parameters
    parser.add_argument('--result_dir', default='./results',
                      help='Result save directory')
    
    return parser.parse_args()

def init_knowledge_graph(db_path, kg_name, dis_function="cosine"):
    """
    Initialize knowledge graph
    
    Args:
        db_path (str): Database path
        kg_name (str): Knowledge graph name
        dis_function (str): Distance function type
        
    Returns:
        tuple: (db_path, collection) Database path and collection object
    """
    print("==========================Start-Initialize Knowledge Graph=========================")
    logger.info(f"Initializing knowledge graph: {kg_name}...")
    db_path, collection = init_db(db_path, kg_name, dis_function)
    
    # Check if the knowledge graph is empty
    try:
        count = collection.count()
        logger.info(f"{count} triples already exist in the knowledge graph")
    except Exception as e:
        logger.error(f"Error checking knowledge graph: {e}")
    print("==========================End-Initialize Knowledge Graph=========================")
    return db_path, collection

def get_dataset_info(dataset_path):
    """
    Get dataset information
    
    Args:
        dataset_path (str): Dataset path
        
    Returns:
        str: Dataset source
    """
    try:
        print("==========================Start-Load Dataset Info=========================")
        dataset = DatasetLoader(dataset_path)
        logger.info(f"Successfully loaded dataset: {os.path.basename(dataset_path)}")
        logger.info(f"Dataset source: {dataset.get_source()}")
        logger.info(f"Dataset type: {dataset.get_type()}")
        logger.info(f"Dataset size: {dataset.get_size()} questions")
        print("==========================End-Load Dataset Info=========================")
        return dataset.get_source()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None

def setup_result_file(args):
    """
    Set up result file path and create necessary directories
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        str: Full path to the result file
    """
    # Generate result file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"EC-spLLM_pip_{args.dataset}_{args.mode}_{args.model}_{args.test_num}_R{args.random_seed}_{args.sim_threshold}_{timestamp}.jsonl"
    sub_path=f"EC-spLLM_{args.dataset}"
    subsub_path=f"Random{args.random_seed}"
    result_folder=os.path.join(args.result_dir, sub_path, subsub_path)
    result_path = os.path.join(result_folder, result_filename)
    
    # Ensure result directory exists
    os.makedirs(result_folder, exist_ok=True)
    
    return result_path


def main(): 
    """
    Main function
    """
    # Parse command line arguments
    args = parse_arguments()
    selected_dataset_path=""
    if args.mode == "rephrase":
        selected_dataset_path = rephrase_dataset_path[args.dataset]
    elif args.mode == "ori":
        selected_dataset_path = dataset_path[args.dataset]
    elif args.mode == "mini":
        selected_dataset_path = mini_file_path[args.dataset]
    else:
        logger.error("Unknown mode")
        return
    # Get dataset information
    dataset_source = get_dataset_info(selected_dataset_path)
    if not dataset_source:
        logger.error("Failed to load dataset, program exiting")
        return
    

    db_path, collection = init_knowledge_graph(args.db_path, args.kg_name, args.dis_function)
    result_path = setup_result_file(args)
    start_time = time.time()
    EC_spLLM_pip(args, selected_dataset_path, collection, result_path, db_path, args.openai_api_keys)
    

    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info("\n========== Pipeline execution completed ==========")
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    logger.info(f"Results have been saved to: {result_path}")

if __name__ == "__main__":
    main() 