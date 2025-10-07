import json
from bert_score import score
from llm_utils import llm_create_triple
import time
from db_utils import kg_add_triple_emb

def extract_entity(ext_entity):
    entity_list = []
    if ext_entity is None:
        return entity_list
    else:
        for triple in ext_entity:
            entity_list.append(triple['head'])
            entity_list.append(triple['tail'])
        if len(entity_list) != 0:
            entity_list = list(set(entity_list))
    return entity_list

def extract_entity_from_prune_result(prune_result):
    triple_list = []
    if prune_result is None:
        print("Prune result is empty")
        return triple_list
    else:
        for item in prune_result:
            triple = item['triple']
            triple_list.append(triple)
    return triple_list

def paragraph_similarity(paragraph1, paragraph2):
    P, R, bert_score = score([paragraph1], [paragraph2], lang='en', model_type='distilbert-base-uncased')
    return P.item(), R.item(), bert_score.item()

def choice_answer_judgment(llm_answer, answer):
    llm_answer_lower = str(llm_answer).lower()
    answer_lower = str(answer).lower()
    return answer_lower in llm_answer_lower

def shot_answer_judgment(llm_answer, answer):
    cor_flag = choice_answer_judgment(llm_answer, answer)
    P, R, bert_score = paragraph_similarity(llm_answer, answer)
    return cor_flag, P, R, bert_score

def shot_save_2_jsonl(id, question, real_answer, llm_answer, entropy, is_insufficient_info, reason, metrics_flag, correct_flag, gold_answer_flag, score, execution_time, mean_time, db_cnt, total_entity_relations_list, prune_entity_relations_list, cluster_chain_of_entities, file_name):
    """
    Save short answer results to jsonl file
    """
    result = {
        "id": id,
        "result": correct_flag,
        "metrics_flag": metrics_flag,
        "gold_answer_flag": gold_answer_flag,
        "entropy": entropy,
        "is_insufficient_info": is_insufficient_info,
        "question": question,
        "real_answers": real_answer,
        "llm_answer": llm_answer,
        "reason": reason,
        "score": score,
        "exe_time": execution_time,
        "search_time": mean_time,
        "db_cnt": db_cnt,
        "total_sup_info": total_entity_relations_list,
        "prune_sup_info": prune_entity_relations_list,
        "used_sup_info": cluster_chain_of_entities
    }
    with open(file_name, "a", encoding='utf-8') as outfile:
        json_str = json.dumps(result, ensure_ascii=False)
        outfile.write(json_str + "\n")

def choice_save_2_jsonl(id, question, choice, real_answer, llm_answer, entropy, is_insufficient_info, reason, metrics_flag, correct_flag, gold_answer_flag, score, execution_time, mean_time, db_cnt, total_entity_relations_list, prune_entity_relations_list, cluster_chain_of_entities, file_name):
    """
    Save multiple choice results to jsonl file
    """
    result = {
        "id": id,
        "result": correct_flag,
        "metrics_flag": metrics_flag,
        "gold_answer_flag": gold_answer_flag,
        "entropy": entropy,
        "is_insufficient_info": is_insufficient_info,
        "question": question,
        "choice": choice,
        "real_answers": real_answer,
        "llm_answer": llm_answer,
        "reason": reason,
        "score": score,
        "exe_time": execution_time,
        "search_time": mean_time,
        "db_cnt": db_cnt,
        "total_sup_info": total_entity_relations_list,
        "prune_sup_info": prune_entity_relations_list,
        "used_sup_info": cluster_chain_of_entities
    }
    with open(file_name, "a", encoding='utf-8') as outfile:
        json_str = json.dumps(result, ensure_ascii=False)
        outfile.write(json_str + "\n")

def extract_triple(create_triple_prompt, question, choices, answer, args, db_path, engine="gpt-4.1"):
    fg, tg = 0, 0
    time_list = []
    while (fg == 0 and tg < 100):
        try:
            start_time = time.time()
            new_triple = llm_create_triple(create_triple_prompt, question, choices, answer, args.openai_api_keys, engine=engine)
            time_list.append(time.time() - start_time)
            fg = 1
            tg += 1
        except Exception as e:
            print("create_triple error", e)
            tg += 1
            time.sleep(5)
    for i in range(len(new_triple)):
        try:
            start_time = time.time()
            db_path, kgname, triple, collection = kg_add_triple_emb(db_path, args.kg_name, new_triple[i], question + str(answer))
            time_list.append(time.time() - start_time)
            print("Add knowledge:", new_triple[i])
        except:
            continue
    return time_list

def print_avg_score(a):
    if len(a) == 0:
        return 0, 0, 0
    P = 0
    R = 0
    F1 = 0
    for i in range(len(a)):
        P += a[i][0]
        R += a[i][1]
        F1 += a[i][2]
    return P / len(a), R / len(a), F1 / len(a)

if __name__ == "__main__":
    pass