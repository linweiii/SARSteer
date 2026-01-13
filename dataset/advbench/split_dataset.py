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

file_path = './'
data_name = "advbench_audio_safe"
text_file = os.path.join(file_path, f"{data_name}.json")

with open(text_file, "r") as f:
    data_json = json.load(f)
data_num = len(data_json)
print(f"Total data num: {data_num}")
org_data = data_json

train_data, test_data = train_test_split(org_data, train_size=train_data_num, random_state=42)

test_data_num = len(test_data)
out_file_train = os.path.join(file_path, f"{data_name}_train_{train_data_num}.json")
out_file_test = os.path.join(file_path, f"{data_name}_test_{test_data_num}.json")

with open(out_file_test, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)
with open(out_file_train, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

pass