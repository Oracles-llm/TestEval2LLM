import os
from pathlib import Path


DEFAULT_REPO_ID = "Qwen/Qwen2-0.5B-Instruct"


def get_repo_id() -> str:
    return os.getenv("HF_REPO_ID", DEFAULT_REPO_ID)


def get_model_root() -> Path:
    return Path(os.getenv("MODEL_DIR", "models")).resolve()


def get_gguf_root() -> Path:
    return Path(os.getenv("GGUF_DIR", "gguf")).resolve()


def get_local_model_dir(repo_id: str | None = None) -> Path:
    repo_id = repo_id or get_repo_id()
    return get_model_root() / repo_id.replace("/", "_")


def get_default_gguf_name(repo_id: str | None = None) -> str:
    repo_id = repo_id or get_repo_id()
    return f"{repo_id.replace('/', '_')}.f16.gguf"
