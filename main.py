import os
import argparse
from utils.utils import set_random_seeds, set_logging, make_dir_if_not_exist, write_result
from info import model_to_fullname, data_name_to_path
from clean_tasks import *
from clean_evaluation import *
from jailbreak_tasks import *
from jailbreak_evaluation import *
from loads import load_data
from defense import *

def main(args):

    data = load_data(args.dataset)

    if 1 in args.execute_steps:
        logger.info(f"### Step1: Answering Questions: {args.jailbreak_task}")
        function_name = f"{args.jailbreak_task}"
        if function_name in globals():
            globals()[function_name](args, logger, data)
        else:
            raise ValueError(f"Function {function_name} is not defined.")
            

    if 2 in args.execute_steps:
        if args.eval_model == "string_matching":
            refusal_rate,num4eval = eval_string_matching(args, logger)
            logger.info(f"The refusal rate for {args.jailbreak_task}-{args.dataset}-{args.model} is:\n {refusal_rate}")
            write_result(args.eval_result_file, args.model, 'Refusal_Rate', num4eval, refusal_rate)
        else:
            if 'clean' in args.jailbreak_task or 'airbench' in args.dataset:
                logger.info(f"### Step2: Evaluating Clean Tasks: {args.jailbreak_task}, {args.dataset}")
                clean_function_name = f"eval_{args.dataset}"
                if clean_function_name in globals():
                    globals()[clean_function_name](args, logger)
                else:
                    raise ValueError(f"Clean function {clean_function_name} is not defined.")
            else:
                logger.info(f"### Step2: Jailbreak Evaluation: {args.eval_model}")
                if args.eval_model == "mistral_sorryBench":
                    asr, num4eval = eval_mistral_sorryBench(args, logger)
                else:
                    raise ValueError(f"Evaluation model {args.eval_model} is not supported.")
                logger.info(f"The {args.eval_model} ASR for {args.jailbreak_task}-{args.dataset}-{args.model} is:\n {asr}")
                write_result(args.eval_result_file, args.model, 'ASR', num4eval, asr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2_audio")#, choices=model_to_fullname.keys())
    parser.add_argument("--dataset", type=str, default="figstep_audio_test", choices=data_name_to_path.keys())
    '''
        jailbreak_task defines all tasks (harmful, safe, defensive).
        - Call SARSteer: defense_audio1_sarsteer
        - Call general-purpose tasks, generally specifying the dataset 'airbench_chat': cleanAudio1
        - Call voice harmful tasks: audio1
        - New tasks can be added by yourself (providing plain text, voice text, visual, etc.)
    '''
    parser.add_argument("--jailbreak_task", "-jtask", default="audio1", type=str, choices=["audio1", "cleanAudio1", "defense_audio1_sarsteer"]) 
    parser.add_argument("--eval_model", "-emodel", default="mistral_sorryBench", type=str, choices=["mistral_sorryBench", "string_matching", "gpt"],
                        help="Evaluation model. 'mistral_sorryBench'/'string_matching' for jailbreak ASR; 'gpt' for AIR-Bench clean evaluation.")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--execute_steps", "-exe", default=[1,2], type=int, nargs='+')
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument("--train_harm_dataset", type=str, default="figstep_audio_train", choices=data_name_to_path.keys())
    parser.add_argument("--train_safe_dataset", type=str, default="figstep_audio_safe_train", choices=data_name_to_path.keys())

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_samples_train", type=int, default=100)

    parser.add_argument("--threshold_angle", type=float, default=10)
    parser.add_argument("--protection_width", type=float, default=1.5)
    parser.add_argument("--k_", type=int, default=10)
    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--exp", type=int, default=0, help="Choose from 0 to 3, corresponding to different refusal prompt variants")
    parser.add_argument("--no_filtering", action="store_true", default=False, help="Disable bimodal peak filtering in steering vector computation")

    args = parser.parse_args()
    set_random_seeds(args.seed)
    args.main_task_path = os.path.join(args.save_path, args.jailbreak_task + "-seed" + str(args.seed))
    args.save_path = os.path.join(args.main_task_path, args.dataset +"-"+ args.model)
    make_dir_if_not_exist(args.save_path)
    # if 'clean' in args.jailbreak_task:
    #     args.eval_model = ""
    output_name = f'output-response.json'
    args.output_file = os.path.join(args.save_path, output_name)
    output_eval_name = f'output-eval-{args.eval_model}.json'
    args.output_eval_file = os.path.join(args.save_path, output_eval_name)
    eval_result_name = f'eval-result-{args.dataset}-{args.eval_model}.csv'
    args.eval_result_file = os.path.join(args.main_task_path, eval_result_name)

    logger = set_logging(os.path.join(args.save_path, "logs"))
    logger.info("Arguments: %s", args)

    # if "CUDA_VISIBLE_DEVICES" in os.environ:
    #     logger.info(f"Using CUDA_VISIBLE_DEVICES from environment: {os.environ['CUDA_VISIBLE_DEVICES']}")
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #     logger.info(f"CUDA_VISIBLE_DEVICES not set in environment, defaulting to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    main(args)