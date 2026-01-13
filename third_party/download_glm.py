import os
from huggingface_hub import snapshot_download

os.makedirs("./models", exist_ok=True)

tokenizer_repo = "THUDM/glm-4-voice-tokenizer"
decoder_repo = "THUDM/glm-voice-decoder"

# 下载 tokenizer 仓库到 ./models 下
tokenizer_local_path = snapshot_download(repo_id=tokenizer_repo, cache_dir="./models")
print("Tokenizer downloaded to:", tokenizer_local_path)

# 下载 decoder 仓库到 ./models 下
decoder_local_path = snapshot_download(repo_id=decoder_repo, cache_dir="./models")
print("Decoder downloaded to:", decoder_local_path)
