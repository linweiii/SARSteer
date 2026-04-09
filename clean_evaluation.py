from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import logging
import os
import threading
import time
from func_timeout import func_set_timeout
import requests
import datetime
from tqdm import tqdm
from utils.utils import *

gpt_info = {
    'url': os.environ.get('GPT_API_URL', 'YOUR_ACTUAL_URL'),
    'headers': {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.environ.get("GPT_API_TOKEN", "YOUR_ACTUAL_TOKEN")}'
    }
}

MAX_RESPONSE_LENGTH = 512

def save_response_to_json(output_eval_file, response_d, score):
    response_d["eval_timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response_d["score"] = score
    append_to_json_file(output_eval_file, response_d)

def cal_eval_airbench_chat_score(logger, eval_path):
    eval_file = load_json_file(eval_path)
    score_dict = {}
    count_nums = {}
    for e in eval_file:
        if e["task_name"] not in score_dict:
            score_dict[e["task_name"]] = []
        if e["task_name"] not in count_nums:
            count_nums[e["task_name"]] = 0
        try:
            score_split = e["gen"].split(' ')
            if len(score_split) < 2:
                logger.info(f"Invalid score values in response: {e['gen']}. Skipping this entry.")
                continue
            score1 = int(score_split[0])
            score2 = int(score_split[1])
            score = (score1 + score2) / 2
            score_dict[e["task_name"]].append(score)
            count_nums[e["task_name"]] += 1
        except ValueError:
            # raise ValueError(f"Invalid score values in response: {e['gen']}.")
            logger.info(f"Invalid score values in response: {e['gen']}. Skipping this entry.")
            continue

    avg_scores = {task: round(sum(scores) / len(scores), 2) for task, scores in score_dict.items()}
    avg_score_all = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
    avg_scores['average'] = round(avg_score_all, 2) 
    count_total_num = sum(count_nums.values())
    count_nums['average'] = count_total_num
        
    return avg_scores, count_nums


lock = threading.Lock()
finish_count = 0
failed_count = 0

def eval_airbench_chat(args, logger):
    root = args.save_path
    args.in_file = os.path.join(root, 'batch_run_input.json')
    args.out_file = os.path.join(root, 'batch_run_output.json')
    logger.info("Building test file for evaluation...")
    build_test_file(args, logger)
    logger.info("Test file built successfully. Starting evaluation...")
    run_chat_gen(args, logger)

    avg_scores, count_nums = cal_eval_airbench_chat_score(logger, args.out_file)
    for task, score in avg_scores.items():
        logger.info(f"Task: {task}, Average Score: {score}, valid count: {count_nums.get(task, 0)}")
        write_result(args.eval_result_file, args.model, task, count_nums.get(task, 0), score)

    return avg_scores, count_nums

def build_test_file(args, logger):
    response_data = load_json_file(args.output_file)
    if not response_data:
        raise ValueError(f"No data to evaluate in {args.output_file}.")
    
    is_finished, response_data = check_and_load_the_rest(logger, args.in_file, response_data, check_key="uniq_id")
    if is_finished:
        return
    
    pbar = tqdm(response_data)
    pbar.set_description("Building test file for evaluation")
    for response_d in pbar:
        row = response_d

        system_prompt = ("You are a helpful and precise assistant for checking the quality of the answer.\n"
            "[Detailed Audio Description]\nXAudioX\n[Question]\nXQuestionX\n"
            "[The Start of Assistant 1s Answer]\nXAssistant1X\n[The End of Assistant 1s Answer]\n"
            "[The Start of Assistant 2s Answer]\nXAssistant2X\n[The End of Assistant 2s Answer]\n[System]\n"
            "We would like to request your feedback on the performance of two AI assistants in response to the user question "
            "and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.\n"
            "Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. "
            "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. "
            "Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
            "The two scores are separated by a space." 
            )

        path = row['path']
        question = row['question']
        answer_gt = row['answer_gt']
        task_name = row['task_name']
        dataset_name = row['dataset_name']
        response = row['response'][:MAX_RESPONSE_LENGTH]
        uuid = row['uniq_id']

        if response == None:
            continue

        if row.get('meta_info', None) == None:
            raise ValueError(f"Missing meta_info for entry with uniq_id={uuid}. Cannot proceed with evaluation.")
        else:
            meta_info = row['meta_info']

        content = system_prompt.replace("XAudioX", meta_info).replace("XQuestionX", question).replace("XAssistant1X", answer_gt).replace("XAssistant2X", response)
        tmp_d = {
            'uuid': uuid,
            'openai_args': {
                "messages": [{"role": "user", "content": content}]
            },
            'meta_info': meta_info,
            'path': path,
            'question': question,
            'answer_gt': answer_gt,
            'task_name': task_name,
            'dataset_name': dataset_name,
            'Audio-LLM-response': response,
        }
        if random.random() < 0.5:
            tmp_d['openai_args'].update({"temperature": 2.0})
        append_to_json_file(args.in_file, tmp_d)

def run_chat_gen(args, logger):
    out_file = args.out_file
    in_file = load_json_file(args.in_file)
    is_finished, items = check_and_load_the_rest(logger, out_file, in_file, check_key="uniq_id")
    if is_finished:
        return
    global finish_count
    global failed_count

    total_count = 0
    with ThreadPoolExecutor(max_workers=50) as pool:
        for item in items:
            total_count += 1
            pool.submit(task, item, args)
        while finish_count + failed_count < total_count:
            logger.info(f"total:{total_count}  finish:{finish_count}  failed:{failed_count}")
            time.sleep(10)


def task(data, args, MAX_API_RETRY=5):
    global finish_count
    global failed_count
    openai_args = {
        'model': 'gpt-4',  # "gpt-3.5-turbo" version in gpt-4o-mini, "gpt-4" version in gpt-4o-2024-08-06
        'temperature': 1.0,
        'max_tokens': 1024
    }
    openai_args.update(data['openai_args'])
    for i in range(MAX_API_RETRY):
        try:
            result, prompt_tokens, completion_tokens, finish_reason = get_result_by_request(**openai_args)
            item = deepcopy(data)
            item.update({
                "gen": result,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason
            })
            lock.acquire()
            finish_count += 1
            append_to_json_file(args.out_file, item)
            lock.release()
            return

        except Exception as e:
            logging.warning("request error: %s", e)
            pass

    lock.acquire()
    failed_count += 1
    lock.release()

@func_set_timeout(1200)
def get_result_by_request(**kwargs):
    res = requests.post(gpt_info['url'], headers=gpt_info['headers'], data=json.dumps(kwargs))
    response = res.json()
    if res.status_code == 200:
        result = response['choices'][0]['message']['content']
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        finish_reason = response['choices'][0]['finish_reason']
        return result, prompt_tokens, completion_tokens, finish_reason

    else:
        raise Exception(response['messages'])
     

