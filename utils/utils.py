import os
import random
import numpy as np
import torch
import logging
import datetime
import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  

def make_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def set_logging(log_dir):
    make_dir_if_not_exist(log_dir)
    now = datetime.datetime.now()
    log_filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) 

    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setLevel(logging.INFO)  

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def initialize_json_file(file_path):
    """
    Initializes a JSON file at the specified path. If the file already exists, it will be overwritten with an empty list.
    """
    make_dir_if_not_exist(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        json.dump([], f, indent=4)

def append_to_json_file(file_path, new_data):
    """
    Appends new data to a JSON file. If the file does not exist, it will be initialized with the new data.
    """
    if not os.path.exists(file_path):
        initialize_json_file(file_path)

    with open(file_path, "r") as f:
        existing_data = json.load(f)

    if not isinstance(existing_data, list):
        raise ValueError("JSON file content must be a list to append new data.")

    existing_data.append(new_data)

    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

def check_and_load_the_rest(logger, file_path, data, check_key="question"):
    # Check for existing data in the output file and load it
    if os.path.exists(file_path):
        logger.info(f"Loading existing data from {file_path}")
        existing_data = load_json_file(file_path)
        if len(existing_data) >= len(data):
            logger.info("All data has already been processed. Exiting.")
            return True, data
        processed_questions = {entry[check_key] for entry in existing_data}
        data = [d for d in data if d[check_key] not in processed_questions]
    return False, data

def write_result(record_path, model, metric, num4eval, score):
    if not os.path.exists(record_path):
        with open(record_path, 'w') as f:
            f.write('datetime \t model \t metric \t num4eval \t score\n')
    with open(record_path, 'a') as f:
        f.write(f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} \t {model} \t {metric} \t {num4eval} \t {score}\n')


# The function to save the response to a JSON file, different format for different tasks
def save_response_to_json(output_file, scenario, question, response, **kwargs):
    if response is None:
        print(f"Warning: No response generated for scenario: {scenario}, question: {question}. Skipping save.")
        return
    now = datetime.datetime.now()
    result = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": scenario,
        "question": question,
        "response": response,
    }
    # Add any additional keyword arguments to the result
    for key, value in kwargs.items():
        result[key] = value
    # Save the results
    append_to_json_file(output_file, result)

def redirect_savepath(args, folder_name):
    task_path = os.path.join(args.save_path, folder_name)
    make_dir_if_not_exist(task_path)
    args.save_path = task_path
    output_name = f'output-response.json'
    args.output_file = os.path.join(task_path, output_name)
    output_eval_name = f'output-eval-{args.eval_model}.json'
    args.output_eval_file = os.path.join(task_path, output_eval_name)
    eval_result_name = f'eval-result-{args.dataset}-{args.eval_model}.csv'
    args.eval_result_file = os.path.join(task_path, eval_result_name)
    return task_path

def crop_audio(inputs_ids, processor):
    audio_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|audio_bos|>')
    audio_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|audio_eos|>')
    pos = inputs_ids.index(audio_start_token_id) + 1
    pos_end = inputs_ids.index(audio_end_token_id)
    return pos, pos_end

from transformers import pipeline
import torchaudio
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",  # 可以选择 small, medium, large 等版本
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def audio_to_text(audio_tensor, sample_rate=16000):
    """
    将音频tensor转换为文本
    """
    # 确保音频是单声道
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.mean(dim=0)  # 立体声转单声道
    
    # 转换为numpy数组并确保采样率正确
    audio_np = audio_tensor.cpu().numpy()
    
    # 如果采样率不是16000，需要重采样
    if sample_rate != 16000:
        audio_np = torchaudio.functional.resample(
            torch.tensor(audio_np), sample_rate, 16000
        ).numpy()
    
    # 语音识别
    result = asr_pipe(audio_np, generate_kwargs={"language": "english"})
    return result["text"]