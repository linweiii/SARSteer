import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

fix_seed(42)
train_data_num = 100


# file_path = './figstep_audio'
safe_file_path = './figstep_audio_safe'
safe_text_file = os.path.join(safe_file_path, 'audio_files.txt')
harm_file_path = './figstep_audio'
harm_text_file = os.path.join(harm_file_path, 'audio_files.txt')

org_data_harm = []
with open(harm_text_file, "r") as f:
    for line in f:
        audio_file, question = line.strip().split(" ", 1)
        name_split = audio_file.split("_")
        scenario = "_".join(name_split[:-1])
        org_data_harm.append({
            "scenario": scenario,
            "audio_file": audio_file,
            "question": question.replace("-", " "),
        })

org_data_safe = []
with open(safe_text_file, "r") as f:
    for line in f:
        audio_file, question = line.strip().split(" ", 1)
        name_split = audio_file.split("_")
        scenario = "_".join(name_split[:-1])
        org_data_safe.append({
            "scenario": scenario,
            "audio_file": audio_file,
            "question": question.replace("-", " "),
        })

data_num_harm = len(org_data_harm)
print(f"Total harm data num: {data_num_harm}")

train_data_harm, test_data_harm = train_test_split(org_data_harm, train_size=train_data_num, random_state=42)

data_num_safe = len(org_data_safe)
print(f"Total safe data num: {data_num_safe}")

index_list = list(range(data_num_harm))
random.shuffle(index_list)
train_index_harm, test_index_harm = train_test_split(index_list, train_size=train_data_num, random_state=42)
# train_index_safe, test_index_safe = train_test_split(index_list, train_size=train_data_num, random_state=42)

train_data_harm = []
test_data_harm = []
train_data_safe = []
test_data_safe = []
for i in train_index_harm:
    train_data_harm.append(org_data_harm[i])
    train_data_safe.append(org_data_safe[i])
for i in test_index_harm:
    test_data_harm.append(org_data_harm[i])
    test_data_safe.append(org_data_safe[i])


test_data_num = len(test_index_harm)

def save_json(file_path, train_data, test_data):
    out_file_train = os.path.join(file_path, f"train_{train_data_num}.json")
    out_file_test = os.path.join(file_path, f"test_{test_data_num}.json")

    with open(out_file_test, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    with open(out_file_train, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

save_json(harm_file_path, train_data_harm, test_data_harm)
save_json(safe_file_path, train_data_safe, test_data_safe)
pass