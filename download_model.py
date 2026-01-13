import os
from pathlib import Path

from huggingface_hub import snapshot_download

from config import get_local_model_dir, get_model_root, get_repo_id


def main() -> None:
    repo_id = get_repo_id()
    model_root = get_model_root()
    model_root.mkdir(parents=True, exist_ok=True)
    local_dir = get_local_model_dir(repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)

    force_download = (
        os.getenv("FORCE_DOWNLOAD", "").lower() in {"1", "true", "yes"}
    )
    has_local_files = (local_dir / "config.json").exists() and any(
        local_dir.glob("*.safetensors")
    )
    if force_download or not has_local_files:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

    print(f"Model downloaded: {local_dir}")


if __name__ == "__main__":
    main()
