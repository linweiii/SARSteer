
from huggingface_hub import snapshot_download

# 下载数据集
snapshot_download(
    repo_id="qyang1021/AIR-Bench-Dataset",  # 例如 "m-a-p/COIG-CQIA"
    repo_type="dataset",
    local_dir="",  # 例如 "./data/test"
    resume_download=True  # 支持断点续传
)