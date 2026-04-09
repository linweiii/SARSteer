import os
import json
import csv
import sys
from info import data_name_to_path, model_to_fullname
import base64
from openai import OpenAI
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration
    from qwen_omni_utils import process_mm_info
    from qwen_vl_utils import process_vision_info
except (ImportError, ModuleNotFoundError):
    pass

import librosa
import torch
from transformers import AutoConfig,TextIteratorStreamer
try:
    from third_party.kimia_infer.api.kimia import KimiAudio
except (ImportError, ModuleNotFoundError):
    pass
from scipy.stats import gaussian_kde
import copy
from tqdm import tqdm
import soundfile as sf
from utils.utils import *

'''
This script is used to load data from different datasets.
The "scenario", "audio_path", and "question" are required for the subsequent processing.
'''
def load_data(dataset):
    if dataset == "figstep_audio" or dataset == "figstep_audio_safe":
        audio_path = os.path.join(data_name_to_path[dataset], "question")
        text_path = os.path.join(data_name_to_path[dataset], "audio_files.txt")
        data = []
        with open(text_path, "r") as f:
            for line in f:
                audio_file, question = line.strip().split(" ", 1)
                name_split = audio_file.split("_")
                scenario = "_".join(name_split[:-1])
                data.append({
                    "scenario": scenario,
                    "audio_path": os.path.join(audio_path, audio_file),
                    "question": question.replace("-", " "),
                })
    elif dataset == "figstep_audio_train" or dataset == "figstep_audio_safe_train":
        audio_path = os.path.join(data_name_to_path[dataset], "question")
        text_path = os.path.join(data_name_to_path[dataset], "train_100.json")
        with open(text_path, "r") as f:
            data_json = json.load(f)
        data = []
        for d in data_json:
            scenario = d["scenario"]
            audio_file = d["audio_file"]
            question = d["question"]
            data.append({
                "scenario": scenario,
                "audio_path": os.path.join(audio_path, audio_file),
                "question": question,
            })
    elif dataset == "figstep_audio_test" or dataset == "figstep_audio_safe_test":
        audio_path = os.path.join(data_name_to_path[dataset], "question")
        text_path = os.path.join(data_name_to_path[dataset], "test_250.json")
        with open(text_path, "r") as f:
            data_json = json.load(f)
        data = []
        for d in data_json:
            scenario = d["scenario"]
            audio_file = d["audio_file"]
            question = d["question"]
            data.append({
                "scenario": scenario,
                "audio_path": os.path.join(audio_path, audio_file),
                "question": question,
            })

    elif dataset == "sorrybench_base":
        audio_path = os.path.join(data_name_to_path[dataset], "base_question_audio")
        text_path = os.path.join(data_name_to_path[dataset], "base_question.jsonl")
        data = []
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                dline = json.loads(line.strip()) 
                question_id = dline["question_id"]
                data.append({
                    "scenario": dline["category"],
                    "audio_path": os.path.join(audio_path, f"{question_id}.wav"),
                    "question": dline["turns"][0],
                })
    elif dataset == "advbench_audio":
        audio_path = os.path.join(data_name_to_path[dataset], "advbench_audio")
        text_path = os.path.join(data_name_to_path[dataset], "advbench_audio.json")
        with open(text_path, "r") as f:
            data_json = json.load(f)
        data = []
        for d in data_json:
            question_id = d["question_id"]
            audio_name = d["audio_name"]
            question = d["question"]
            data.append({
                "scenario": "harmful_behaviors",
                "audio_path": os.path.join(audio_path, audio_name),
                "question": question,
            })
    elif dataset == "advbench_audio_safe":
        audio_path = os.path.join(data_name_to_path[dataset], "advbench_audio_safe")
        text_path = os.path.join(data_name_to_path[dataset], "advbench_audio_safe.json")
        with open(text_path, "r") as f:
            data_json = json.load(f)
        data = []
        for d in data_json:
            question_id = d["question_id"]
            audio_name = d["audio_name"]
            question = d["question"]
            data.append({
                "scenario": "safe_behaviors",
                "audio_path": os.path.join(audio_path, audio_name),
                "question": question,
            })

    elif dataset == "airbench_chat":
        input_file = os.path.join(data_name_to_path[dataset], "Chat_meta.json")
        data = []

        with open(input_file, "r") as fin:
            fin = json.load(fin)
            for item in fin:
                wav = item['path']
                task_name = item['task_name']
                dataset_name = item['dataset_name']
                audio_path = wav_fn = f'{data_name_to_path[dataset]}/{task_name}_{dataset_name}/{wav}'
                if os.path.exists(wav_fn) == False:
                    print(f"lack wav {wav_fn}")
                    continue
                data.append({
                    "scenario": f"{task_name}_{dataset_name}",
                    "audio_path": audio_path,
                    "question": item['question'],
                    "item": item,
                })

    elif dataset == "ajailbench_base":
        audio_path = data_name_to_path[dataset]
        data = []
        for root, dirs, files in os.walk(audio_path): 
            for file in files:
                if file.endswith(".wav"):
                    wav_fn = os.path.join(audio_path, file)
                    data.append({
                        "scenario": file.split(".")[0],
                        "audio_path": wav_fn,
                        "question": wav_fn,
                    })
        

    # visual datasets
    elif dataset == "figstep_visual":
        image_path = os.path.join(data_name_to_path[dataset], "SafeBench")
        text_path = os.path.join(data_name_to_path[dataset], "safebench.csv")
        data = []

        with open(text_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                scenario = row[3]
                image_file = os.path.join(image_path, f"query_{row[0]}_{row[1]}_{row[2]}_6.png")
                question = row[4]
                data.append({
                    "scenario": scenario,
                    "image_path": image_file,
                    "question": question,
                })
    elif dataset == "mm_vet":
        image_path = os.path.join(data_name_to_path[dataset], "extracted_images")
        text_path = os.path.join(data_name_to_path[dataset], "extracted_text.csv")
        data = []
        with open(text_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario = row['capability']
                image_file = row['image_path']
                question = row['question']
                data.append({
                    "scenario": scenario,
                    "image_path": os.path.join(image_path, image_file),
                    "question": question,
                })
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented.")
    return data


'''
To load different models.
'''
def load_model(model_name, **kwargs):
    if model_name == "gpt4o_audio":
        return gpt4o_audio(model=model_to_fullname[model_name], **kwargs)
    elif model_name == "qwen2_audio":
        return qwen2_audio(model=model_to_fullname[model_name], **kwargs)
    elif model_name == "qwen2_audio_base":
        return qwen2_audio(model=model_to_fullname[model_name], **kwargs)
    elif model_name == "qwen_audio":
        return qwen_audio(model=model_to_fullname[model_name], **kwargs)
    elif model_name == "qwen2_5Omni":
        return qwen2_5Omni(model=model_to_fullname[model_name], **kwargs)
    elif model_name == "qwen2":
        return qwen2(model=model_to_fullname[model_name], **kwargs)
    elif model_name == "kimi_audio":
        return kimi_audio(model=model_to_fullname[model_name], **kwargs)

    # VLM
    elif model_name == "qwen2_5vl":
        return qwen2_5vl(model=model_to_fullname[model_name], **kwargs)
    
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented.")


class gpt4o_audio:
    def __init__(self, api_key=None, model="gpt-4o-audio-preview", audio_format="wav"):
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.audio_format = audio_format
        self.model = model
        # self.output_modalities = ["text"]

    def encode_audio(self, audio_path):
        # Load the audio file
        with open(audio_path, "rb") as audio_file:
            wav_data = audio_file.read()

        # Convert the audio data to a base64 encoded string
        encoded_string = base64.b64encode(wav_data).decode('utf-8')
        return encoded_string
    
    def messages_template(self, input_text, audio_path):
        if audio_path is None:
            messages=[
                {
                    "role": "user",
                    "content": [
                        { 
                            "type": "text",
                            "text": input_text
                        }
                    ]
                },
            ]
        else:
            messages=[
                {
                    "role": "user",
                    "content": [
                        { 
                            "type": "text",
                            "text": input_text
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": self.encode_audio(audio_path),
                                "format": self.audio_format
                            }
                        }
                    ]
                },
            ]
        return messages
    
    def generate_response(self, input_text, audio_path):
        # if self.is_output_audio:
        if audio_path is None:
            response = self.client.chat.completions.create(
                model=self.model,
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "wav"},
                messages=self.messages_template(input_text, audio_path)
            )
            # print(response)
            text_content = response.choices[0].message.audio.transcript
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                modalities=["text"],
                messages=self.messages_template(input_text, audio_path)
            )
            text_content = response.choices[0].message.content
        return text_content
    
class qwen2_audio:
    def __init__(self, model="Qwen/Qwen2-Audio-7B-Instruct", sampling_rate=16000, system_prompt=None, hook_hidden_states=False, hook_one_sample=True, hook_all_token=False):
        self.system_prompt = system_prompt
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.processor.feature_extractor.sampling_rate = sampling_rate
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model, device_map="auto")
        
        self.language_model = self.model.language_model.model
        self.layer_num = len(self.model.language_model.model.layers)
        self.device = self.language_model.device
        if hook_hidden_states:
            self.hook_one_sample=hook_one_sample
            self.hook_all_token=hook_all_token
            self.hidden_states = {}
            self.hook_hidden_states()

    def messages_template(self, input_text, audio_path):
        conversation = []
        if self.system_prompt is not None:
            conversation.append({"role": "system", "content": self.system_prompt})
        if audio_path is None:
            conversation.append({"role": "user", "content": input_text})
        else:
            conversation.append({"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": input_text},
            ]})
        return conversation

    def prepare_input(self, input_text, audio_path=None):
        conversation = self.messages_template(input_text, audio_path)
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        if audio_path is None:
            inputs = self.processor.tokenizer(text, return_tensors="pt", padding=True)
        else:
            # Load the audio file
            audio, _ = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
            # Prepare the input for the model
            inputs = self.processor(
                text=text,
                audio=audio,
                return_tensors="pt",
                padding=True,
            )
        return inputs

    def generate_response(self, input_text, audio_path=None):
        inputs = self.prepare_input(input_text, audio_path)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Generate the response
        generate_ids = self.model.generate(**inputs, max_new_tokens=512)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        # Decode the generated response
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response
    
    def hook_hidden_states(self):
        for layer_idx in range(self.layer_num):
            def hook_fn(module, input, output, idx=layer_idx):
                if self.hook_all_token:
                    out = copy.copy(output.squeeze().detach().cpu())
                else:
                    out = copy.copy(output[:,-1,:].squeeze().detach().cpu())
                if self.hook_one_sample:
                    self.hidden_states[idx] = out
                else:
                    if self.hidden_states.get(idx) is None:
                        self.hidden_states[idx] = []
                    self.hidden_states[idx].append(out) # last token hidden states
                return output
            self.language_model.layers[layer_idx].register_forward_hook(hook_fn)

    def release_hook(self):
        for layer_idx in range(self.layer_num):
            self.language_model.layers[layer_idx]._forward_hooks.clear()


class qwen_audio:
    def __init__(self, model="Qwen/Qwen-Audio-Chat"):
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="cuda", trust_remote_code=True).eval()
        self.device = self.model.device
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.language_model = self.model.transformer.h
        self.layer_num = len(self.language_model)
    
    def prepare_input(self, input_text, audio_path):
        if audio_path is None:
            query = self.tokenizer.from_list_format([
                {'text': input_text},
            ])
        else:
            query = self.tokenizer.from_list_format([
                {'text': input_text},
                {'audio': audio_path}, # Either a local path or an url
            ])
        return query

    def generate_response(self, input_text, audio_path, max_new_tokens=-1):
        query = self.prepare_input(input_text, audio_path)
        if max_new_tokens > 0:
            response, history = self.model.chat(self.tokenizer, query=query, history=None, max_new_tokens=max_new_tokens+1)
        else:
            response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response
    
USE_AUDIO_IN_VIDEO=False
class qwen2_5Omni:
    def __init__(self, model="Qwen/Qwen2.5-Omni-7B", sampling_rate=16000, system_prompt=None, hook_hidden_states=False, hook_one_sample=True, hook_all_token=False):
        # self.system_prompt = system_prompt
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model, trust_remote_code=True)
        # self.processor.feature_extractor.sampling_rate = sampling_rate
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model, device_map="auto", torch_dtype="auto", trust_remote_code=True)

        self.language_model = self.model.thinker.model
        self.layer_num = len(self.model.thinker.model.layers)
        self.device = self.language_model.device
        if hook_hidden_states:
            self.hook_one_sample=hook_one_sample
            self.hook_all_token=hook_all_token
            self.hidden_states = {}
            self.hook_hidden_states()

    def messages_template(self, input_text, audio_path):
        conversation = []
        # if self.system_prompt is not None:
        conversation.append({
            "role": "system",
            "content": [
                {"type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        })
        if audio_path is None:
            conversation.append({"role": "user", "content": [{"type": "text", "text": input_text}]})
        else:
            conversation.append({"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": input_text},
            ]})
        return conversation

    def prepare_input(self, input_text, audio_path=None):
        conversation = self.messages_template(input_text, audio_path)
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        if audio_path is None:
            inputs = self.processor.tokenizer(text, return_tensors="pt", padding=True)
        else:
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        return inputs

    def generate_response(self, input_text, audio_path=None):
        inputs = self.prepare_input(input_text, audio_path)
        inputs = inputs.to(self.model.device).to(self.model.dtype) 
        # Generate the response
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        text_output = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = text_output[0]
        start = response.rfind('assistant\n')+len('assistant\n')
        response = response[start:]
        return response
    
    def hook_hidden_states(self):
        for layer_idx in range(self.layer_num):
            def hook_fn(module, input, output, idx=layer_idx):
                if self.hook_all_token:
                    out = copy.copy(output.squeeze().detach().cpu())
                else:
                    out = copy.copy(output[:,-1,:].squeeze().detach().cpu())
                if self.hook_one_sample:
                    self.hidden_states[idx] = out
                else:
                    if self.hidden_states.get(idx) is None:
                        self.hidden_states[idx] = []
                    self.hidden_states[idx].append(out) # last token hidden states
                return output
            self.language_model.layers[layer_idx].register_forward_hook(hook_fn)
    
class qwen2:
    def __init__(self, model="Qwen/Qwen2-7B-Instruct", system_prompt=None, hook_hidden_states=False):
        self.system_prompt = system_prompt
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.language_model = self.model.model
        self.layer_num = len(self.language_model.layers)
        self.device = self.language_model.device
        if hook_hidden_states:
            self.hidden_states = {}
            self.hook_hidden_states()

    def messages_template(self, input_text, audio_path=None):
        conversation = []
        if self.system_prompt is not None:
            conversation.append({"role": "system", "content": self.system_prompt})
        conversation.append({"role": "user", "content": input_text})
        return conversation
    
    def prepare_input(self, input_text, audio_path=None):
        conversation = self.messages_template(input_text, audio_path)
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return model_inputs
    
    def generate_response(self, input_text, audio_path=None):
        model_inputs = self.prepare_input(input_text)

        generated_ids = self.model.generate(
            model_inputs.input_ids
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
    
    def hook_hidden_states(self):
        for layer_idx in range(self.layer_num):
            def hook_fn(module, input, output, idx=layer_idx):
                if self.hidden_states.get(idx) is None:
                    self.hidden_states[idx] = []
                self.hidden_states[idx].append(copy.copy(output[:,-1,:].squeeze().detach().cpu())) # last token hidden states
                return output
            self.language_model.layers[layer_idx].register_forward_hook(hook_fn)

    def release_hook(self):
        for layer_idx in range(self.layer_num):
            self.language_model.layers[layer_idx]._forward_hooks.clear()
    
class kimi_audio:
    def __init__(self, model="moonshotai/Kimi-Audio-7B-Instruct", system_prompt=None, hook_hidden_states=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        self.model = KimiAudio(model_path=model, load_detokenizer=True)
        self.sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
        }
        self.layer_num = len(self.model.alm.model.layers)
        self.language_model = self.model.alm.model
        self.device = self.language_model.device
        if hook_hidden_states:
            self.hidden_states = {}
            self.hook_hidden_states()

    def messages_template(self, input_text, audio_path):
        conversation = []
        if self.system_prompt is not None:
            conversation.append({"role": "assistant", "message_type": "text", "content": self.system_prompt})

        if audio_path is None:
            conversation.append({"role": "user", "message_type": "text", "content": input_text})
        else:
            conversation.append({"role": "user", "message_type": "audio", "content": audio_path})
            conversation.append({"role": "user", "message_type": "text", "content": input_text})
        return conversation
    
    def prepare_input(self, input_text, audio_path):
        conversation = self.messages_template(input_text, audio_path)
        return conversation
    
    def _get_empty_audio(self):
        """Generate a short silent audio file for cases with no audio input."""
        empty_audio_path = os.path.join(os.path.dirname(__file__), '.cache', 'empty_audio.wav')
        if not os.path.exists(empty_audio_path):
            os.makedirs(os.path.dirname(empty_audio_path), exist_ok=True)
            import numpy as np
            sr = 16000
            silence = np.zeros(sr, dtype=np.float32)  # 1 second of silence
            sf.write(empty_audio_path, silence, sr)
        return empty_audio_path

    def generate_response(self, input_text, audio_path=None, max_new_tokens=-1):
        if audio_path is None:
            audio_path = self._get_empty_audio()
            print("No audio input, use empty audio instead.")
        else:
            audio, sr = librosa.load(audio_path)
            if (audio == 0).all():
                audio_path = self._get_empty_audio()
                print("Zero length audio input, use empty audio instead.")
        model_inputs = self.prepare_input(input_text, audio_path)

        _, response = self.model.generate(
            model_inputs,
            **self.sampling_params,  # Unpack sampling parameters
            output_type="text",
            max_new_tokens=max_new_tokens
        )

        return response

    def hook_hidden_states(self):
        for layer_idx in range(self.layer_num):
            def hook_fn(module, input, output, idx=layer_idx):
                if self.hidden_states.get(idx) is None:
                    self.hidden_states[idx] = []
                self.hidden_states[idx].append(copy.copy(output[0][:,-1,:].squeeze().detach().cpu())) # last token hidden states
                return output
            self.model.alm.model.layers[layer_idx].register_forward_hook(hook_fn)

    def release_hook(self):
        for layer_idx in range(self.layer_num):
            self.language_model.layers[layer_idx]._forward_hooks.clear()


## VLM
class qwen2_5vl:
    def __init__(self, model="Qwen/Qwen2-5-VL-7B-Instruct", system_prompt=None, hook_hidden_states=False):
        self.system_prompt = system_prompt
        # self.processor = Qwen2_5VLProcessor.from_pretrained(model, trust_remote_code=True)
        # self.model = Qwen2_5VLModel.from_pretrained(model, device_map="auto", trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.layer_num = len(self.model.language_model.layers)
        self.language_model = self.model.language_model
        self.device = self.language_model.device
        if hook_hidden_states:
            self.hidden_states = {}
            self.hook_hidden_states()

    def messages_template(self, input_text, image_path):
        conversation = []
        if self.system_prompt is not None:
            conversation.append({"role": "system", "content": self.system_prompt})
        if image_path is None:
            conversation.append({"role": "user", "content": input_text})
        else:
            conversation.append({"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": input_text},
            ]})
        return conversation

    def prepare_input(self, input_text, image_path=None):
        conversation = self.messages_template(input_text, image_path)
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        if image_path is None:
            inputs = self.processor.tokenizer(text, return_tensors="pt", padding=True)
        else:
            # Load the image file
            # image = Image.open(image_path).convert("RGB")
            image_input, video_input = process_vision_info(conversation)
            # Prepare the input for the model
            inputs = self.processor(
                text=[text],
                images=image_input,
                videos=video_input,
                return_tensors="pt",
                padding=True,
            )
        return inputs

    def generate_response(self, input_text, image_path=None):
        inputs = self.prepare_input(input_text, image_path)
        inputs = inputs.to(self.model.device).to(self.model.dtype) 
        # Generate the response
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
    
    def hook_hidden_states(self):
        for layer_idx in range(self.layer_num):
            def hook_fn(module, input, output, idx=layer_idx):
                if self.hidden_states.get(idx) is None:
                    self.hidden_states[idx] = []
                self.hidden_states[idx].append(copy.copy(output[0][:,-1,:].squeeze().detach().cpu())) # last token hidden states
                return output
            self.model.language_model.layers[layer_idx].register_forward_hook(hook_fn)

