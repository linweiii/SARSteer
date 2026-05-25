from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",  
    repo_type="model", 
    local_dir="./ft-mistral-7b-instruct-v0.2-sorry-bench-202406",  
    local_dir_use_symlinks=False,  
    resume_download=True, 
    token=""  
)