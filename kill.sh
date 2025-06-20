ps -ef | grep preprocess_senorita.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep inference_vie_score_filter.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep run.py | grep -v grep | awk '{print $2}' | xargs kill -9



from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="SENORITADATASET/Senorita",
    repo_type="dataset",
    revision="main",
    allow_patterns=["inpainting_upload/*"],
    local_dir=".",             # 会在当前目录生成 inpainting_upload/…
    local_dir_use_symlinks=False, 
    resume_download=True
)
