import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    repo_id = os.getenv("HF_REPO_ID", "Qwen/Qwen2.5-3B-Instruct")
    model_root = Path(os.getenv("MODEL_DIR", "models")).resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    local_dir = model_root / repo_id.replace("/", "_")
    local_dir.mkdir(parents=True, exist_ok=True)

    force_download = os.getenv("FORCE_DOWNLOAD", "").lower() in {"1", "true", "yes"}
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
