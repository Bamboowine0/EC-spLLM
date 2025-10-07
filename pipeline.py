import numpy as np
from tqdm import tqdm
import time
import os
from llm_utils import *
from prompt_manager import PromptManager
from utils import *
from db_utils import query_db_search, db_count
from data.dataset_loader import DatasetLoader
import random
from sentence_transformers import SentenceTransformer
import concurrent.futures
import threading
import time

total_score_list = []

id = 0
id0 = id
model_path = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_path)

metrics_dict = {
    0: "Pass",
    1: "Insufficient Information",
    2: "Semantic Entropy Exceeds Threshold"
}

def EC_spLLM_pip(args, dataset_path, collection, result_path, db_path, openai_api_keys):
    """
    Main pipeline function for EC-spLLM

    :param args: Command line arguments
    :param dataset_path: Path to the dataset, pointing to a unified format JSON file
    :param collection: Knowledge graph collection
    :param result_path: Path to save results
    :param db_path: Database path
    :param openai_api_keys: OpenAI API keys
    """
    global total_score_list
    global id
    global id0

    prompt_manager = PromptManager(args.dataset)

    try:
        print("==========================Start-Loading Dataset=========================")
        dataset = DatasetLoader(dataset_path)
        print(f"Successfully loaded dataset: {dataset_path}")
        print(f"Dataset source: {dataset.get_source()}")
        print(f"Dataset size: {dataset.get_size()} questions")
        print(f"Dataset type: {dataset.get_type()}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure the dataset has been converted to the unified format using dataset_converter.py")
        return

    is_multiple_choice = dataset.is_multiple_choice()
    print(f"Is multiple choice: {'Yes' if is_multiple_choice else 'No'}")

    dataset_size = min(args.test_num, dataset.get_size() - id)
    indices = list(range(id, id + dataset_size))
    random.seed(args.random_seed)
    random.shuffle(indices)
    print(f"Processing {dataset_size} questions starting from index {id} (shuffled)")
    print("==========================End-Loading Dataset=========================")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    correct_counter = 0
    gold_answer_counter = 0

    entity_extract_time = []
    search_db_time = []
    prune_time = []
    llm_reason_time = []
    sample_time = []
    cluster_time = []
    self_cognition_time = []
    se_cognition_time = []
    llm_answer_time = []
    extra_knowledge_time = []
    add_knowledge_time = []

    for idx in tqdm(indices):
        id += 1
        start_time = time.time()
        early_stop_flag = False
        gold_answer_flag = False

        try:
            question = dataset.get_question(idx)
            answer = dataset.get_answer(idx)
            choices = dataset.get_choices(idx) if is_multiple_choice else None

            print(f"\nProcessing question {id}/{id0 + dataset_size}:")
            print(f"Question: {question}")
            if choices:
                print(f"Choices: {choices}")

            prune_entity_relations_list = {}
            total_entity_relations_list = {}
            entity_list = []
            retrieve_result_after_prune = []
            entity_filter = []
            search_db_time = []

            entity_list_q = llm_recgn_entity(prompt_manager, question, args.openai_api_keys, engine=args.model)
            entity_extract_time.append(time.time() - start_time)
            print(f"entity_extract_time:{entity_extract_time[-1]}")

            for depth in range(1, args.depth + 1):
                print("*" * 25)
                print(f"\nStart knowledge retrieval at layer {depth}...")
                print("*" * 25)

                current_entity_relations_list = {}
                correct_flag = None
                metrics_flag = None

                if depth == 1:
                    entity_list = entity_list_q
                else:
                    entity_list = extract_entity(retrieve_result_after_prune)

                new_entity_list = [entity for entity in entity_list if entity not in entity_filter]
                print(f"Number of entities to retrieve at this layer: {len(new_entity_list)}")

                start_time = time.time()
                print(f"Start parallel retrieval of {len(new_entity_list)} entities...")

                if len(new_entity_list) > 0:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(new_entity_list), 8)) as executor:
                        future_to_entity = {
                            executor.submit(parallel_entity_search, question, entity, collection,
                                            args.width, args.dist, args.sim_threshold, args.sim_top, model): entity
                            for entity in new_entity_list
                        }

                        for future in concurrent.futures.as_completed(future_to_entity):
                            entity, retrieve_result, once_search_time = future.result()

                            if retrieve_result is not None:
                                current_entity_relations_list[entity] = retrieve_result
                else:
                    print("No new entities to retrieve, skipping parallel retrieval step")
                search_db_time.append(time.time() - start_time)
                print(f"search_db_time:{search_db_time[-1]}")

                print("==========================Start-Knowledge Pruning=========================")
                start_time = time.time()
                if len(current_entity_relations_list) > 0:
                    if not is_multiple_choice:
                        retrieve_with_scores = llm_relation_prune(prompt_manager, current_entity_relations_list, question, args.openai_api_keys, args.width, engine=args.model)
                    else:
                        retrieve_with_scores = llm_relation_prune(prompt_manager, current_entity_relations_list, question, args.openai_api_keys, args.width, choice=choices, engine=args.model)
                else:
                    retrieve_with_scores = []

                for item in retrieve_with_scores:
                    item['triple'] = fix_triple_order(item['triple'], current_entity_relations_list)

                filtered_triples = []
                for item in retrieve_with_scores:
                    if check_triple_in_original(item['triple'], current_entity_relations_list):
                        filtered_triples.append(item)
                    else:
                        print(f"Remove hallucinated triple: {item['triple']}")

                retrieve_with_scores = filtered_triples
                retrieve_result_after_prune = extract_entity_from_prune_result(retrieve_with_scores)

                if len(retrieve_result_after_prune) == 0:
                    early_stop_flag = True

                entity_filter.extend(new_entity_list)
                if entity_filter:
                    entity_filter = list(set(entity_filter))

                prune_entity_relations_list[depth] = retrieve_with_scores
                total_entity_relations_list[depth] = current_entity_relations_list

                print("Knowledge extracted after pruning:")
                print(prune_entity_relations_list)
                print("==========================End-Knowledge Pruning=========================")
                end_time = time.time()
                prune_time.append(end_time - start_time)
                print(f"prune_time:{prune_time[-1]}")

                llm_answer, llm_reason, support_kg_triple, entropy, is_insufficient_info, time_list = llm_reason_with_kg(
                    prompt_manager, question, prune_entity_relations_list, args.openai_api_keys, choices, engine=args.model)
                llm_reason_time.append(time_list[0])
                print(f"llm_reason_time:{llm_reason_time[-1]}")
                sample_time.append(time_list[1])
                print(f"sample_time:{sample_time[-1]}")
                cluster_time.append(time_list[2])
                print(f"cluster_time:{cluster_time[-1]}")
                self_cognition_time.append(time_list[3])
                print(f"self_cognition_time:{self_cognition_time[-1]}")
                se_cognition_time.append(time_list[4])
                print(f"se_cognition_time:{se_cognition_time[-1]}")

                if is_multiple_choice:
                    metrics_flag = distinguish_metrics(is_insufficient_info, entropy)
                    if metrics_flag == 0 or early_stop_flag:
                        if early_stop_flag:
                            print(f"!!!!!!!!!!!!!!!!!")
                            print(f"No new knowledge retrieved this round, early stopping")
                            print(f"!!!!!!!!!!!!!!!!!")
                        correct_flag = evaluate_multiple_choice(llm_answer, answer)
                        if correct_flag:
                            correct_counter += 1
                        gold_answer_flag = is_gold_answer(metrics_flag, depth)
                        save_result(is_multiple_choice, metrics_flag, correct_flag, gold_answer_flag, id, question, choices, answer, llm_answer, entropy, is_insufficient_info, llm_reason,
                                    total_entity_relations_list, prune_entity_relations_list, support_kg_triple,
                                    result_path, start_time, search_db_time, db_path, args)
                        break
                else:
                    metrics_flag = distinguish_metrics(is_insufficient_info, entropy)
                    if metrics_flag == 0 or early_stop_flag:
                        if early_stop_flag:
                            print(f"!!!!!!!!!!!!!!!!!")
                            print(f"No new knowledge retrieved this round, early stopping")
                            print(f"!!!!!!!!!!!!!!!!!")
                        start_time = time.time()
                        correct_flag = evaluate_short_answer(question, llm_answer, answer)
                        llm_answer_time.append(time.time() - start_time)
                        print(f"llm_answer_time:{llm_answer_time[-1]}")
                        if correct_flag:
                            correct_counter += 1
                        gold_answer_flag = is_gold_answer(metrics_flag, depth)
                        save_result(is_multiple_choice, metrics_flag, correct_flag, gold_answer_flag, id, question, choices, answer, llm_answer, entropy, is_insufficient_info, llm_reason,
                                    total_entity_relations_list, prune_entity_relations_list, support_kg_triple,
                                    result_path, start_time, search_db_time, db_path, args)
                        break

                if depth == args.depth and metrics_flag != 0:
                    if is_multiple_choice:
                        correct_flag = evaluate_multiple_choice(llm_answer, answer)
                        if correct_flag:
                            correct_counter += 1
                        gold_answer_flag = is_gold_answer(metrics_flag, depth)
                        save_result(is_multiple_choice, metrics_flag, correct_flag, gold_answer_flag, id, question, choices, answer, llm_answer, entropy, is_insufficient_info, llm_reason,
                                    total_entity_relations_list, prune_entity_relations_list, support_kg_triple,
                                    result_path, start_time, search_db_time, db_path, args)
                    else:
                        start_time = time.time()
                        correct_flag = evaluate_short_answer(question, llm_answer, answer)
                        llm_answer_time.append(time.time() - start_time)
                        print(f"llm_answer_time:{llm_answer_time[-1]}")
                        if correct_flag:
                            correct_counter += 1
                        gold_answer_flag = is_gold_answer(metrics_flag, depth)
                        save_result(is_multiple_choice, metrics_flag, correct_flag, gold_answer_flag, id, question, choices, answer, llm_answer, entropy, is_insufficient_info, llm_reason,
                                    total_entity_relations_list, prune_entity_relations_list, support_kg_triple,
                                    result_path, start_time, search_db_time, db_path, args)

            final_start_time = time.time()
            g_counter, add_knowledge_time_list = handle_insufficient_knowledge(metrics_flag, is_multiple_choice, question, llm_answer, answer, choices,
                                                                              db_path, args, depth, prompt_manager, engine=args.model, gold_answer_flag=gold_answer_flag)
            gold_answer_counter += g_counter
            extra_knowledge_time.append(add_knowledge_time_list[0])
            print(f"extra_knowledge_time:{extra_knowledge_time[-1]}")
            add_knowledge_time.append(add_knowledge_time_list[1])
            print(f"add_knowledge_time:{add_knowledge_time[-1]}")

            print("--------------------------------")
            print(f"gold_answer_counter: {gold_answer_counter}")
            print(f"gold_answer ratio: {gold_answer_counter / id}")
            print("--------------------------------")
            add_knowledge_time.append(time.time() - final_start_time)
        except Exception as e:
            print(f"Error processing question: {e}")
            continue

    print_final_results(is_multiple_choice, total_score_list, correct_counter, dataset_size, args)
    print(f"Entity extraction time: {np.mean(entity_extract_time)}")
    print(f"Retrieval time: {np.mean(search_db_time)}")
    print(f"Pruning time: {np.mean(prune_time)}")
    print(f"Reasoning time: {np.mean(llm_reason_time)}")
    print(f"Sampling time: {np.mean(sample_time)}")
    print(f"Clustering time: {np.mean(cluster_time)}")
    print(f"Self-cognition time: {np.mean(self_cognition_time)}")
    print(f"Semantic entropy cognition time: {np.mean(se_cognition_time)}")
    print(f"Answer judgment time: {np.mean(llm_answer_time)}")
    print(f"Knowledge extraction time: {np.mean(extra_knowledge_time)}")
    print(f"Knowledge addition time: {np.mean(add_knowledge_time)}")

def distinguish_metrics(is_insufficient_info, entropy):
    judge_flag = 0
    print(f"**********Start-Metric Judgment**********")
    if is_insufficient_info:
        judge_flag = 1
    elif entropy > 0:
        judge_flag = 2
    print(f"Judgment result: {metrics_dict[judge_flag]}")
    print(f"**********End-Metric Judgment**********")
    return judge_flag

def evaluate_multiple_choice(llm_answer, answer):
    correct_flag = choice_answer_judgment(llm_answer, answer)

    print("**************Start-Multiple Choice Answer Evaluation**************")
    print("LLM Answer:")
    print(llm_answer)
    print(f"Correct answer: {answer}")
    print(f"Answer {'Correct' if correct_flag else 'Incorrect'}")
    print("**************End-Multiple Choice Answer Evaluation**************")

    return correct_flag

def evaluate_short_answer(question, llm_answer, answer):
    correct_text = run_llm_answer_judge(question, llm_answer, answer)
    correct_flag = json.loads(correct_text).get("result")

    print("**************Start-Short Answer Evaluation**************")
    print("LLM Answer:")
    print(llm_answer)
    print(f"Answer {'Correct' if correct_flag else 'Incorrect'}")
    print("**************End-Short Answer Evaluation**************")

    return correct_flag

def save_result(is_multiple_choice, metrics_flag, result, gold_answer_flag, id, question, choices, answer, llm_answer, entropy, is_insufficient_info, llm_reason,
                total_relations, prune_relations, support_triples, result_path, start_time,
                search_times, db_path, args, metrics="NaN"):
    end_time = time.time()
    execution_time = end_time - start_time
    mean_time = np.mean(search_times) if search_times else 0
    db_cnt = db_count(db_path, args.kg_name)

    if is_multiple_choice:
        choice_save_2_jsonl(
            id, question, choices, answer, llm_answer, entropy, is_insufficient_info, llm_reason, metrics_flag, result, gold_answer_flag, metrics,
            execution_time, mean_time, db_cnt, total_relations, prune_relations,
            support_triples, result_path
        )
    else:
        shot_save_2_jsonl(
            id, question, answer, llm_answer, entropy, is_insufficient_info, llm_reason, metrics_flag, result, gold_answer_flag, metrics,
            execution_time, mean_time, db_cnt, total_relations, prune_relations,
            support_triples, result_path
        )

def handle_insufficient_knowledge(metrics_flag, is_multiple_choice, question, llm_answer, answer, choices,
                                  db_path, args, depth, prompt_manager, engine="gpt-4.1", gold_answer_flag=False):
    is_gold_answer = 0
    print(f"**********Start-Knowledge Addition**********")
    if metrics_flag:
        print(f"{metrics_dict[metrics_flag]} occurred, using gold answer")
        time_list = add_knowledge_to_db(question, answer, choices, args, db_path, is_multiple_choice, prompt_manager, engine=engine)
        is_gold_answer = 1
    else:
        if depth >= 2:
            print(f"Search is costly (depth > 2), using gold answer")
            time_list = add_knowledge_to_db(question, answer, choices, args, db_path, is_multiple_choice, prompt_manager, engine=engine)
            is_gold_answer = 1
        else:
            print(f"Answer metrics normal, not using gold answer")
            time_list = add_knowledge_to_db(question, llm_answer, choices, args, db_path, is_multiple_choice, prompt_manager, gold_answer=0, engine=engine)
            is_gold_answer = 0
    print(f"**********End-Knowledge Addition**********")
    return is_gold_answer, time_list

def is_gold_answer(metrics_flag, depth):
    if metrics_flag:
        return True
    elif depth >= 2:
        return True
    else:
        return False

def add_knowledge_to_db(question, answer, choices, args, db_path, is_multiple_choice, prompt_manager, gold_answer=1, engine="gpt-4.1"):
    print("Adding knowledge to knowledge graph...")

    try:
        if is_multiple_choice:
            if gold_answer:
                time_list = extract_triple(prompt_manager, question, choices, choices[int(answer)], args, db_path, engine=engine)
            else:
                if args.dataset == "pubmed_qa":
                    time_list = extract_triple(prompt_manager, question, choices, answer, args, db_path, engine=engine)
                else:
                    time_list = extract_triple(prompt_manager, question, choices, choices[int(answer)], args, db_path, engine=engine)
        else:
            time_list = extract_triple(prompt_manager, question, choices, answer, args, db_path, engine=engine)

        print("Knowledge added successfully")
    except Exception as e:
        print(f"Error adding knowledge: {e}")
    return time_list

def print_final_results(is_multiple_choice, score_list, correct_count, total_count, args):
    print("\n========== Processing Complete ==========")

    if not is_multiple_choice:
        print(f"Dataset {args.dataset} scores: P: {avg_P:.4f}, R: {avg_R:.4f}, F1: {avg_F1:.4f}")
    else:
        acc = correct_count / total_count if total_count > 0 else 0
        print(f"Dataset {args.dataset} accuracy: {acc:.4f} ({correct_count}/{total_count})")


def parallel_entity_search(question, entity, collection, width, dist, sim_threshold, sim_top, model):
    """
    Parallel retrieval of knowledge for a single entity

    Args:
        question (str): Query question
        entity (str): Entity to retrieve
        collection: ChromaDB collection object
        width (int): Maximum number of results to return
        dist (float): Distance threshold
        sim_threshold (float): Similarity threshold
        sim_top (int): Top N triples with highest similarity to return
        model: SentenceTransformer model

    Returns:
        tuple: (entity, retrieve_result, search_time)
    """
    start_search_time = time.time()
    print(f"Retrieving entity: {entity}")

    retrieve_result = query_db_search(question, entity, collection, width, dist, sim_threshold, sim_top, model)

    end_search_time = time.time()
    once_search_time = end_search_time - start_search_time

    if retrieve_result is not None:
        print(f"Found {len(retrieve_result)} related knowledge")
    else:
        print(f"!!! No related knowledge found !!!")

    return entity, retrieve_result, once_search_time
