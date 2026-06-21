from huggingface_hub import snapshot_download
snapshot_download(repo_id="prashantpanchal058/recommendation-models", local_dir="src/model_artifacts")

# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="prashantpanchal058/recommendation-models", local_dir="src/model_artifacts")