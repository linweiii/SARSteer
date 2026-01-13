import os
from tqdm import tqdm
import random

audio_path = 'AJailbreak_Base'  # Replace with your actual path
target_folder = 'AJailbreak_Base_subset'
os.makedirs(target_folder, exist_ok=True)

files = [os.path.join(root, file) for root, dirs, files in os.walk(audio_path) for file in files if file.endswith(".wav")]
random.shuffle(files)
files_subset = files[:200]
for file in tqdm(files_subset, desc="Copying audio files"):
    fn = os.path.basename(file)
    target_fn = os.path.join(target_folder, fn)
    with open(file, 'rb') as f_in:
        with open(target_fn, 'wb') as f_out:
            f_out.write(f_in.read())
