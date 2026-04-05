from huggingface_hub import snapshot_download
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    "rain1011/pyramid-flow-sd3",
    local_dir="/data/pyramid-flow-sd3/",
    local_dir_use_symlinks=False,
    repo_type='model'
)
#model_path = '/data/pyramid-flow-miniflux'   # The local directory to save downloaded checkpoint
#snapshot_download("rain1011/pyramid-flow-miniflux", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')

