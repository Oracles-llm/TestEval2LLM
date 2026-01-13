import os
from pathlib import Path

from llama_cpp import Llama


def pick_gguf_file(gguf_dir: Path, preferred_file: str | None) -> Path:
    if preferred_file:
        gguf_file = Path(preferred_file)
        candidate = gguf_dir / gguf_file.name
        if candidate.exists():
            return candidate
        raise RuntimeError(f"GGUF file not found locally: {candidate}")

    gguf_files = [p for p in gguf_dir.glob("*.gguf")]
    if not gguf_files:
        raise RuntimeError(
            "No .gguf files found. Run make_gguf.py after downloading the model."
        )

    for suffix in ("Q4_K_M.gguf", "Q4_K.gguf", "Q5_K_M.gguf", "Q5_K.gguf"):
        for f in gguf_files:
            if f.name.endswith(suffix):
                return f

    return gguf_files[0]


def build_llm(model_path: Path) -> Llama:
    return Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=max(os.cpu_count() or 2, 2),
        n_gpu_layers=int(os.getenv("N_GPU_LAYERS", "0")),
    )


def main() -> None:
    repo_id = os.getenv("HF_REPO_ID", "Qwen/Qwen2.5-3B-Instruct")
    gguf_root = Path(os.getenv("GGUF_DIR", "gguf")).resolve()
    preferred_file = os.getenv("HF_MODEL_FILE")
    model_path = pick_gguf_file(gguf_root, preferred_file)
    llm = build_llm(model_path)

    prompt = os.getenv("PROMPT")
    if prompt:
        response = llm(prompt, max_tokens=128, stop=["</s>"])
        print(response["choices"][0]["text"].strip())
        return

    print("Model ready. Type a prompt and press enter. Type 'exit' to quit.")
    while True:
        user_prompt = input(">>> ").strip()
        if not user_prompt or user_prompt.lower() in {"exit", "quit"}:
            break
        response = llm(user_prompt, max_tokens=256, stop=["</s>"])
        print(response["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
