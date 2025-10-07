from openai import OpenAI
import time
import json
import re
import random
from prompt_manager import PromptManager
from prompt import clustering_prompt, answer_judege_prompt, choice_cluster_prompt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
pattern = re.compile(r'Content is blocked$')

router = "openai"
opeani_api_keys = {
    "openai": "your key",
    "ds": "your key"
}
base_url = {
    "openai": "https://api.openai.com/v1",
    "ds": "https://api.deepseek.com"
}

def run_llm(prompt, opeani_api_key="", temperature=0.1, max_tokens=2048, engine="gpt-4.1"):
    f, t = 0, 0
    while f == 0 and t < 100:
        try:
            if engine == "deepseek-chat":
                router = 'ds'
                client = OpenAI(
                    api_key=opeani_api_keys[router],
                    base_url=base_url[router]
                )
            else:
                client = OpenAI(
                    api_key=opeani_api_keys[router],
                    base_url=base_url[router]
                )
            response = client.chat.completions.create(
                model=engine,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content
            result = pattern.sub('', result)
            f = 1
            t += 1
        except Exception as e:
            print("openai error:", e)
            t += 1
            time.sleep(3)
    return result

def run_llm_answer_judge(question, llm_answer, real_answer):
    print("***********Start-openai answer judge***********")
    prompt_text = answer_judege_prompt.format(q=question, llm_answer=llm_answer, real_answer=real_answer)
    prompt = [
        {"role": "user", "content": prompt_text}
    ]
    result = run_llm(prompt, opeani_api_keys[router], engine="gpt-4.1")
    print(result)
    print("***********End-openai answer judge***********")
    return result

def run_llm_logprobs_threaded(prompt, opeani_api_key="", temperature=1, max_tokens=4096, engine="gpt-4.1"):
    print("start openai logprobs (threaded)")
    answer_data_list = {
        "answer": [],
        "data": [],
        "logprobs": []
    }

    def single_query(thread_id):
        f, t = 0, 0
        while f == 0 and t < 100:
            try:
                if engine == "deepseek-chat":
                    router = 'ds'
                    client = OpenAI(
                        api_key=opeani_api_keys[router],
                        base_url=base_url[router]
                    )
                else:
                    client = OpenAI(
                        api_key=opeani_api_keys[router],
                        base_url=base_url[router]
                    )
                response = client.chat.completions.create(
                    model=engine,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    logprobs=True,
                    top_logprobs=1,
                    n=1
                )
                content_text = response.choices[0].message.content
                try:
                    content_json = json.loads(content_text)
                    answer_start = -1
                    answer_end = -1
                    in_answer = False
                    answer_token_count = 0
                    for j, token_info in enumerate(response.choices[0].logprobs.content):
                        token = token_info.token
                        if 'answer' in token and (j + 1 < len(response.choices[0].logprobs.content) and ":" in response.choices[0].logprobs.content[j + 1].token):
                            answer_start = j + 2
                            in_answer = True
                        elif in_answer:
                            if '\",\n' in token or '\"\n' in token or '.\"' in token or ',\"' in token or '\"}' in token or '\",' in token:
                                answer_end = j
                                in_answer = False
                                break
                            else:
                                answer_token_count += 1
                    if answer_start != -1 and answer_end != -1:
                        answer_sum = 0
                        if answer_end - answer_start > 1:
                            for j in range(answer_start + 1, answer_end):
                                answer_sum += response.choices[0].logprobs.content[j].logprob
                            answer_token_count = answer_token_count - 2
                            answer_sum_trans = np.exp(answer_sum / answer_token_count)
                        else:
                            for j in range(answer_start, answer_end + 1):
                                answer_sum += response.choices[0].logprobs.content[j].logprob
                            answer_sum_trans = np.exp(answer_sum / answer_token_count)
                        print(f"Total logprob of answer field: {answer_sum}")
                        print(f"Total probability of answer field (after conversion): {answer_sum_trans}")
                        print(f"Token count of answer field: {answer_token_count}")
                        print(f"")
                        try:
                            data = json.loads(content_text)
                            answer = data.get("answer", "Parsing failed")
                        except:
                            answer = "Parsing failed"
                            t = t + 1
                            time.sleep(2)
                            continue
                        print(f"++++++++Start-Thread {thread_id} finished++++++++")
                        print(f"answer: {answer}")
                        print(f"logprobs: {answer_sum_trans}")
                        print(f"++++++++End-Thread {thread_id} finished++++++++")
                        return {
                            "logprobs": answer_sum_trans,
                            "data": content_text,
                            "answer": answer
                        }
                    else:
                        print("Could not find answer field boundaries")
                        print(response.choices[0].logprobs.content)
                        t += 1
                        time.sleep(2)
                        continue
                except Exception as e:
                    print("JSON parsing failed:", e)
                    print("Failed content_text:")
                    print(content_text)
                    print("Exception occurred at line:", e.__traceback__.tb_lineno)
                    t += 1
                    time.sleep(2)
                    continue
            except Exception as e:
                print("openai error:", e)
                t += 1
                time.sleep(2)
                continue
        return 123

    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(single_query, i) for i in range(10)]
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                if type(res["answer"]) != str:
                    res["answer"] = str(res["answer"])
                results.append(res)
    end_time = time.time()
    print(f"Total execution time for ten threads: {end_time - start_time:.2f} seconds")

    if not results:
        print("No valid results obtained from all threads!")

    for res in results:
        answer_data_list["logprobs"].append(res["logprobs"])
        answer_data_list["data"].append(res["data"])
        answer_data_list["answer"].append(res["answer"])

    print("answer_data_list:")
    print(answer_data_list)

    return answer_data_list

def answer_clustering(mes_pair_list: list, question: str, choice=None, engine: str = "gpt-4.1") -> list:
    """Cluster the answers in the answer list"""
    answer_list = []
    for i in range(len(mes_pair_list)):
        answer_list.append(list(mes_pair_list[i].keys()))
    if engine == "deepseek-chat":
        client = OpenAI(
            api_key=opeani_api_keys["ds"],
            base_url=base_url["ds"]
        )
    else:
        client = OpenAI(
            api_key=opeani_api_keys[router],
            base_url=base_url[router]
        )
    if engine == "deepseek-chat" and choice is not None:
        prompt = [{
            "role": "system",
            "content": choice_cluster_prompt["system"]
        },
        {
            "role": "user",
            "content": choice_cluster_prompt["user"].format(sentences=answer_list)
        }]
    else:
        prompt = [{
            "role": "system",
            "content": clustering_prompt["system"]
        },
        {
            "role": "user",
            "content": clustering_prompt["user"].format(sentences=answer_list, q=question)
        }]
    response = client.chat.completions.create(
        model=engine,
        messages=prompt,
        response_format={"type": "json_object"},
    )
    result = response.choices[0].message.content
    print(result)
    return result

def answer_clustering_forexp(answer_data_list: dict, question: str, choice=None, engine: str = "gpt-4.1") -> list:
    """Cluster based on answers and logprobs in answer_data_list"""
    answer_list = answer_data_list["answer"]
    logprobs_list = answer_data_list["logprobs"]
    mes_pair_list = []
    for i in range(len(answer_list)):
        mes_pair_list.append({answer_list[i]: logprobs_list[i]})
    result = answer_clustering(mes_pair_list, question, choice=choice, engine=engine)
    return result

def insufficient_cluster_judgment(prompt_manager: PromptManager, cluster_data: dict, engine: str = "gpt-4.1") -> list:
    """Determine if information is insufficient"""
    if engine == "deepseek-chat":
        client = OpenAI(
            api_key=opeani_api_keys["ds"],
            base_url=base_url["ds"]
        )
    else:
        client = OpenAI(
            api_key=opeani_api_keys[router],
            base_url=base_url[router]
        )
    prompt = [{
        "role": "system",
        "content": prompt_manager.get_insufficient_info_prompt()
    },
    {
        "role": "user",
        "content": json.dumps(cluster_data)
    }]
    response = client.chat.completions.create(
        model=engine,
        messages=prompt,
        response_format={"type": "json_object"},
    )
    result = response.choices[0].message.content
    print("**************Start-Insufficient Information Judgment**************")
    print(result)
    print("**************End-Insufficient Information Judgment**************")
    return result

def semantic_entropy_forexp(prompt_manager: PromptManager, prompt, question, choice: str = None, knowledge_list: list = None, openai_api_keys="", temperature=1, max_tokens=4096, engine="gpt-4.1") -> int:
    print("==========================Start-Calculate Entropy=========================")
    hdea_time_list = []
    sample_time_start = time.time()
    answer_data_list = run_llm_logprobs_threaded(prompt=prompt, opeani_api_key=openai_api_keys, temperature=1, max_tokens=max_tokens, engine=engine)
    hdea_time_list.append(time.time() - sample_time_start)
    f, t = 0, 0
    while f == 0 and t < 10:
        try:
            start_time = time.time()
            mes_cluster = answer_clustering_forexp(answer_data_list, question, choice=choice, engine=engine)
            end_time = time.time()
            hdea_time_list.append(end_time - start_time)
            print(f"Clustering time: {end_time - start_time:.2f} seconds")
            clusters = json.loads(mes_cluster).get("clusters", None)
            if clusters is None:
                print("Could not find clusters field, retrying clustering")
                t += 1
                time.sleep(2)
            else:
                if not clusters:
                    print("!!!!!!!!!!!!!!!!clusters is empty, retrying clustering!!!!!!!!!!!!!!!!")
                    t += 1
                    time.sleep(2)
                    continue
                re_cluster = False
                for i in range(len(clusters)):
                    for j in range(len(clusters[i][list(clusters[i].keys())[0]])):
                        if clusters[i][list(clusters[i].keys())[0]][j] not in answer_data_list["answer"]:
                            print("Hallucinated answer found during clustering:")
                            print(clusters[i])
                            print("Retrying clustering")
                            print("--------------------------------")
                            re_cluster = True
                            break
                    if re_cluster:
                        break
                if re_cluster:
                    t += 1
                    time.sleep(2)
                    continue
                f = 1
                t += 1
        except Exception as e:
            print(f"Error parsing clustering result: {e}, retrying clustering")
            t += 1
            time.sleep(2)

    f, t = 0, 0
    while f == 0 and t < 10:
        try:
            start_time = time.time()
            if choice is None:
                insufficient_info_cluster = insufficient_cluster_judgment(prompt_manager, json.loads(mes_cluster), engine=engine)
            else:
                insufficient_info_cluster = """{"insufficient_info": []}"""
            end_time = time.time()
            hdea_time_list.append(end_time - start_time)
            print(f"Insufficient information judgment time: {end_time - start_time:.2f} seconds")
            if insufficient_info_cluster is None:
                print("Could not find insufficient_info result, retrying judgment")
                t += 1
                time.sleep(2)
            else:
                f = 1
                t += 1
        except Exception as e:
            print(f"Error parsing insufficient_info result: {e}, retrying judgment")
            t += 1
            time.sleep(2)
    insufficient_info = json.loads(insufficient_info_cluster).get("insufficient_info", [])
    print("insufficient_info:")
    print(insufficient_info)

    pos_sum = 0
    for i in range(len(answer_data_list["logprobs"])):
        pos_sum += answer_data_list["logprobs"][i]
    print("**************Start-Probability Sum**************")
    print(pos_sum)
    print("**************End-Probability Sum**************")

    start_time = time.time()
    cluster_sum = []
    normalized_cluster_sum = []
    insufficient_info_entropy = 0
    is_insufficient_info = False
    for i in range(len(clusters)):
        cluster_sum.append(0)
        for j in range(len(clusters[i][list(clusters[i].keys())[0]])):
            for k in range(len(answer_data_list["answer"])):
                if clusters[i][list(clusters[i].keys())[0]][j] == answer_data_list["answer"][k]:
                    cluster_sum[i] += answer_data_list["logprobs"][k]
                    answer_data_list["logprobs"].pop(k)
                    answer_data_list["answer"].pop(k)
                    answer_data_list["data"].pop(k)
                    break
        normalized_cluster_sum.append(cluster_sum[i] / pos_sum)
        if i + 1 in insufficient_info:
            insufficient_info_entropy += normalized_cluster_sum[-1]
    print("**************Start-Cluster Probability Sum & Insufficient Info Probability Sum**************")
    print("normalized_cluster_sum:")
    print(normalized_cluster_sum)
    print("insufficient_info_entropy:")
    print(insufficient_info_entropy)
    print("**************End-Cluster Probability Sum & Insufficient Info Probability Sum**************")
    if insufficient_info_entropy > 0.5:
        is_insufficient_info = True
    if choice is not None and not is_insufficient_info:
        if engine != "deepseek-chat":
            is_insufficient_info = llm_confidence_judgment(prompt_manager, question, knowledge_list, openai_api_keys, choice=choice, engine=engine)

    entropy = 0
    for i in range(len(clusters)):
        if normalized_cluster_sum[i] == 0:
            continue
        entropy += normalized_cluster_sum[i] * np.log(normalized_cluster_sum[i])
    entropy = -entropy
    print("**************Start-Entropy**************")
    print(entropy)
    print("**************End-Entropy**************")
    print("**************Start-Insufficient Information Judgment**************")
    print(is_insufficient_info)
    print("**************End-Insufficient Information Judgment**************")
    print("==========================End-Calculate Entropy=========================")
    end_time = time.time()
    hdea_time_list.append(end_time - start_time)
    return entropy, is_insufficient_info, hdea_time_list

def llm_recgn_entity(prompt_manager: PromptManager, question: str, openai_api_keys: str,
                     temperature: float = 0.2, max_tokens: int = 2048,
                     engine: str = "gpt-4.1") -> list:
    prompt = [
        {"role": "system", "content": prompt_manager.get_entity_recognition_prompt()["character"]},
        {"role": "user", "content": prompt_manager.format_entity_prompt(question)}
    ]
    result = run_llm(prompt, openai_api_keys, engine=engine)
    print("**************Start-Entity Recognition Result**************")
    print(result)
    print("**************End-Entity Recognition Result**************")
    if 'content' in json.loads(result):
        return json.loads(result)['content']["entities"]
    return json.loads(result)["entities"]

def llm_relation_score(prompt_manager: PromptManager, retrieve_result: str, question: str,
                      openai_api_keys: str, temperature: float = 0.2,
                      max_tokens: int = 2048, engine: str = "gpt-4.1", choice=None) -> list:
    if choice is None:
        prompt = [
            {"role": "system", "content": prompt_manager.get_relation_scoring_prompt()["character"]},
            {"role": "user", "content": prompt_manager.format_relation_prompt(question, retrieve_result=retrieve_result)}
        ]
    else:
        prompt = [
            {"role": "system", "content": prompt_manager.get_relation_scoring_prompt()["character"]},
            {"role": "user", "content": prompt_manager.format_relation_prompt(question, retrieve_result=retrieve_result, choice=choice)}
        ]
    result = run_llm(prompt, openai_api_keys, engine=engine)
    print("**************Start-Model Pruning Result**************")
    print(result)
    print("**************End-Model Pruning Result**************")
    if 'content' in json.loads(result):
        return json.loads(result)['content']["triples"]
    return json.loads(result)["triples"]

def check_triple_format(triple):
    required_fields = ['triple', 'score']
    if not all(field in triple for field in required_fields):
        return False
    if not isinstance(triple['score'], (int, float)):
        return False
    if not isinstance(triple['triple'], dict):
        return False
    triple_fields = ['head', 'relation', 'tail']
    if not all(field in triple['triple'] for field in triple_fields):
        return False
    return True

def llm_relation_prune(prompt_manager: PromptManager, retrieve_result: str, question: str,
                      openai_api_keys: str, width: int = 5, temperature: float = 0.2,
                      max_tokens: int = 2048, engine: str = "gpt-4.1", choice=None) -> list:
    f, t = 0, 0
    while f == 0 and t < 10:
        try:
            if choice is None:
                score_list = llm_relation_score(prompt_manager, retrieve_result, question, openai_api_keys, engine=engine)
            else:
                if prompt_manager.dataset_name == "medmc_qa":
                    score_list = llm_relation_score(prompt_manager, retrieve_result, question, openai_api_keys, engine=engine)
                else:
                    score_list = llm_relation_score(prompt_manager, retrieve_result, question, openai_api_keys, choice=choice, engine=engine)
            format_valid = all(check_triple_format(item) for item in score_list)
            if not format_valid:
                print("**************Warning**************")
                print("Returned JSON format is incorrect, re-scoring...")
                print("**************Warning**************")
                t += 1
                time.sleep(2)
            else:
                f = 1
                t += 1
        except Exception as e:
            print(f"Scoring error: {e}, re-scoring")
            t += 1
            time.sleep(2)

    if t == 10:
        print("Scoring failed, returning empty list")
        return []
    score_list = [item for item in score_list if item['score'] > 0.6]
    if len(score_list) <= width:
        print(f"Number of triples is less than {width}, returning all triples, triple count is {len(score_list)}")
        return score_list
    return score_list[:width]

def llm_confidence_judgment(prompt_manager: PromptManager, question: str, total_entity_relations_list: str,
                      openai_api_keys: str, choice: str = None, engine: str = "gpt-4.1") -> tuple:
    prompt = [
        {"role": "system", "content": prompt_manager.get_confidence_judgment_prompt()["character"]},
        {"role": "user", "content": prompt_manager.format_confidence_judgment_prompt(question, choice, total_entity_relations_list)}
    ]

    f, t = 0, 0
    while f == 0 and t < 10:
        try:
            result = run_llm(prompt, openai_api_keys, engine=engine, temperature=0.1)
            result_json = json.loads(result)
            if "answer" in result_json and result_json["answer"].lower() in ["yes", "no"]:
                f = 1
            else:
                print("Returned result format is incorrect, regenerating...")
                print(result)
                t += 1
                time.sleep(2)
        except Exception as e:
            print(f"Error parsing result: {e}, regenerating")
            t += 1
            time.sleep(2)

    if t == 10:
        print("Failed to generate 10 times, default return No")
        return "No"
    print("**********Start-Confidence Judgment**********")
    print(f"Judgment result: {'Pass' if result_json['answer'] == 'Yes' else 'Fail'}")
    print("**********End-Confidence Judgment**********")
    if result_json["answer"] == "Yes":
        return False
    return True

def llm_reason_with_kg(prompt_manager: PromptManager, question: str, total_entity_relations_list: str,
                      openai_api_keys: str, choice: str = None, engine: str = "gpt-4.1") -> tuple:
    triples = []
    time_list = []
    for item in total_entity_relations_list.values():
        triples.extend(item)
    total_entity_relations_list = triples
    if len(triples) == 0:
        total_entity_relations_list = "No related knowledge"
        prompt = [
            {"role": "system", "content": prompt_manager.get_temporary_prompt()["character"]},
            {"role": "user", "content": prompt_manager.format_temporary_prompt(question, choices=choice)}
        ]
    else:
        prompt = [
            {"role": "system", "content": prompt_manager.get_dataset_prompt()["character"]},
            {"role": "user", "content": prompt_manager.format_dataset_prompt(question, knowledge=total_entity_relations_list, choices=choice)}
        ]
    reason_time_start = time.time()
    result = run_llm(prompt, openai_api_keys, temperature=0.3, engine=engine)
    time_list.append(time.time() - reason_time_start)

    entropy, is_insufficient_info, hdea_time_list = semantic_entropy_forexp(prompt_manager, prompt, question, choice, total_entity_relations_list, openai_api_keys, engine=engine)
    time_list.extend(hdea_time_list)
    print("**************Start-Reasoning Result**************")
    print(result)
    print("**************End-Reasoning Result**************")

    return json.loads(result)['answer'], "no reason", "no support info", entropy, is_insufficient_info, time_list

def llm_create_triple(prompt_manager: PromptManager, question: str, choices: list, answer: str,
                     openai_api_keys: str, temperature: float = 0.2,
                     max_tokens: int = 2048, engine: str = "gpt-4.1") -> list:
    prompt = [
        {"role": "system", "content": prompt_manager.get_triple_creation_prompt()["character"]},
        {"role": "user", "content": prompt_manager.format_triple_prompt(question, choices, answer)}
    ]
    result = run_llm(prompt, openai_api_keys, engine=engine)
    print(result)
    if 'content' in json.loads(result):
        return json.loads(result)['content']['triples']
    return json.loads(result)["triples"]

def fix_triple_order(triple, original_relations):
    for entity, relations in original_relations.items():
        for relation in relations:
            if triple['relation'] == relation['relation']:
                if (triple['head'] == relation['tail'] and
                    triple['tail'] == relation['head']):
                    return {
                        'head': relation['head'],
                        'relation': relation['relation'],
                        'tail': relation['tail']
                    }
    return triple

def check_triple_in_original(triple, original_list):
    for entity in original_list.values():
        for t in entity:
            if (triple == t) or (triple['head'] == t['tail'] and
                triple['tail'] == t['head'] and
                triple['relation'] == t['relation']):
                return True
    return False