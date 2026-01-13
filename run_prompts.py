import json
import os
from datetime import datetime, timezone
from pathlib import Path

from llama_cpp import Llama

from config import get_default_gguf_name, get_gguf_root

def load_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        prompts.append(stripped)
    if not prompts:
        raise RuntimeError(f"No prompts found in {path}")
    return prompts


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
    prompts_path = Path(os.getenv("PROMPTS_FILE", "prompts.txt")).resolve()
    output_path = Path(os.getenv("OUTPUT_FILE", "results/responses.jsonl")).resolve()
    gguf_root = get_gguf_root()
    preferred_file = os.getenv("HF_MODEL_FILE", get_default_gguf_name())

    prompts = load_prompts(prompts_path)
    model_path = pick_gguf_file(gguf_root, preferred_file)
    llm = build_llm(model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_tokens = int(os.getenv("MAX_TOKENS", "128"))
    temperature = float(os.getenv("TEMPERATURE", "0.2"))

    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            response = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>"],
            )
            text = response["choices"][0]["text"].strip()
            record = {
                "prompt": prompt,
                "response": text,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model_path": str(model_path),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Saved responses to {output_path}")


if __name__ == "__main__":
    main()
