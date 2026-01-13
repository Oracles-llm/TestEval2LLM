import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def extract_llama_cpp_assets(tools_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if "/gguf-py/" in name or name.endswith("/convert_hf_to_gguf.py"):
                target = tools_dir / "/".join(name.split("/")[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())


def ensure_converter(tools_dir: Path) -> Path:
    tools_dir.mkdir(parents=True, exist_ok=True)
    convert_path = tools_dir / "convert_hf_to_gguf.py"
    gguf_pkg = tools_dir / "gguf-py" / "gguf" / "__init__.py"

    if convert_path.exists() and gguf_pkg.exists():
        return convert_path

    zip_urls = [
        "https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip",
        "https://github.com/ggerganov/llama.cpp/archive/refs/heads/main.zip",
    ]
    last_error = None
    for url in zip_urls:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = Path(tmp) / "llama_cpp.zip"
                urllib.request.urlretrieve(url, zip_path)
                extract_llama_cpp_assets(tools_dir, zip_path)
            if convert_path.exists() and gguf_pkg.exists():
                return convert_path
        except Exception as exc:  # pragma: no cover - network/remote errors
            last_error = exc
    raise RuntimeError(f"Failed to download llama.cpp assets: {last_error}") from last_error


def main() -> None:
    repo_id = os.getenv("HF_REPO_ID", "Qwen/Qwen2.5-3B-Instruct")
    model_root = Path(os.getenv("MODEL_DIR", "models")).resolve()
    local_dir = model_root / repo_id.replace("/", "_")
    if not local_dir.exists():
        raise RuntimeError(f"Model directory not found: {local_dir}")

    gguf_root = Path(os.getenv("GGUF_DIR", "gguf")).resolve()
    gguf_root.mkdir(parents=True, exist_ok=True)
    out_file = gguf_root / f"{repo_id.replace('/', '_')}.f16.gguf"
    force_convert = os.getenv("FORCE_CONVERT", "").lower() in {"1", "true", "yes"}
    if out_file.exists() and not force_convert:
        print(f"GGUF already exists: {out_file}")
        print("Set FORCE_CONVERT=1 to rebuild.")
        return

    tools_dir = gguf_root / "tools"
    convert_path = ensure_converter(tools_dir)
    command = [
        sys.executable,
        str(convert_path),
        str(local_dir),
        "--outfile",
        str(out_file),
        "--outtype",
        "f16",
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "GGUF conversion failed. Make sure torch, transformers, and "
            "sentencepiece are installed."
        ) from exc
    print(f"GGUF ready: {out_file}")


if __name__ == "__main__":
    main()
