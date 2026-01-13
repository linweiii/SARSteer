from tqdm import tqdm
from utils.utils import *  
from info import *
from loads import *


'''
cleanAudio1: clean audio task.
'''
def cleanAudio1(args, logger, data):
    is_finished, data = check_and_load_the_rest(logger, args.output_file, data, check_key="question")
    if is_finished:
        return
    
    model = load_model(args.model)

    pbar = tqdm(data)
    for d in pbar:
        question = d["question"]
        scenario = d["scenario"]
        audio_path = d["audio_path"]
        item = d["item"]
        pbar.set_description(f"Processing scenario: {scenario}, audio: {audio_path}")

        response = model.generate_response(input_text=question, audio_path=audio_path)

        if args.dataset == "airbench_chat":
            others = {
                        "meta_info": item['meta_info'],
                        "answer_gt": item["answer_gt"],
                        "path": item["path"],
                        "task_name": item['task_name'],
                        "dataset_name": item['dataset_name'],
                        "uniq_id": item["uniq_id"],
                    }
        else:
            others = {}
        save_response_to_json(args.output_file, scenario, question, response, **others)
