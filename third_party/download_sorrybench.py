# ensure parent directory is on sys.path so we can import _tmp_info
from pathlib import Path
import sys
_parent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_parent_dir))
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",
    token="your-huggingface-token-here",
    local_dir="./models/sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",
)
