import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from llama_cpp import Llama

from config import get_default_gguf_name, get_gguf_root

DATASETS = [
    {
        "id": "gsm8k_main",
        "dataset": "gsm8k",
        "config": "main",
        "split": "test",
    },
    {
        "id": "ai2_arc_challenge",
        "dataset": "ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
    },
    {
        "id": "ifeval",
        "dataset": "google/IFEval",
        "config": None,
        "split": "train",
    },
    {
        "id": "mmlu_college_cs",
        "dataset": "cais/mmlu",
        "config": "college_computer_science",
        "split": "test",
    },
    {
        "id": "mmlu_elementary_math",
        "dataset": "cais/mmlu",
        "config": "elementary_mathematics",
        "split": "test",
    },
    {
        "id": "bbh_logical_deduction",
        "dataset": "lukaemon/bbh",
        "config": "logical_deduction_five_objects",
        "split": "test",
    },
]


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


def format_choices(choices: Any) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    lines: list[str] = []

    if isinstance(choices, dict):
        labels = choices.get("label") or []
        texts = choices.get("text") or []
        for label, text in zip(labels, texts):
            label_str = str(label).strip()
            text_str = str(text).strip()
            if not label_str:
                continue
            mapping[label_str] = text_str
            lines.append(f"{label_str}. {text_str}")
    elif isinstance(choices, list):
        for idx, choice in enumerate(choices):
            if isinstance(choice, dict):
                label = str(choice.get("label", "")).strip()
                text = str(choice.get("text", "")).strip()
            else:
                label = chr(ord("A") + idx)
                text = str(choice).strip()
            if not label:
                label = chr(ord("A") + idx)
            mapping[label] = text
            lines.append(f"{label}. {text}")

    return "\n".join(lines), mapping


def record_to_prompt(dataset_id: str, record: dict[str, Any]) -> tuple[str, str]:
    if dataset_id == "gsm8k_main":
        prompt = str(record.get("question", "")).strip()
        ground_truth = str(record.get("answer", "")).strip()
        return prompt, ground_truth

    if dataset_id == "ai2_arc_challenge":
        question = str(record.get("question", "")).strip()
        choices_text, mapping = format_choices(record.get("choices"))
        answer_key = str(record.get("answerKey", "")).strip()
        answer_text = mapping.get(answer_key, "")
        ground_truth = (
            f"{answer_key}. {answer_text}".strip(". ").strip()
            if answer_key
            else answer_text
        )
        prompt = (
            "Answer the question by choosing the correct option letter.\n"
            f"Question: {question}\n"
            f"Choices:\n{choices_text}"
        )
        return prompt, ground_truth

    if dataset_id == "ifeval":
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            prompt = str(record.get("input", "")).strip()
        ground_truth = ""
        return prompt, ground_truth

    if dataset_id.startswith("mmlu_"):
        question = str(record.get("question", "")).strip()
        choices = record.get("choices", [])
        choices_text, mapping = format_choices(choices)
        answer_idx = record.get("answer", None)
        answer_key = ""
        if isinstance(answer_idx, int):
            answer_key = chr(ord("A") + answer_idx)
        answer_text = mapping.get(answer_key, "")
        ground_truth = (
            f"{answer_key}. {answer_text}".strip(". ").strip()
            if answer_key
            else answer_text
        )
        prompt = (
            "Answer the question by choosing the correct option letter.\n"
            f"Question: {question}\n"
            f"Choices:\n{choices_text}"
        )
        return prompt, ground_truth

    if dataset_id == "bbh_logical_deduction":
        prompt = str(record.get("input", "")).strip()
        ground_truth = str(record.get("target", "")).strip()
        return prompt, ground_truth

    return "", ""


def load_dataset_records(entry: dict[str, Any]) -> list[dict[str, Any]]:
    ds_name = entry["dataset"]
    cfg = entry.get("config")
    split = entry.get("split")
    if cfg:
        dataset = load_dataset(ds_name, cfg, split=split)
    else:
        dataset = load_dataset(ds_name, split=split)
    return list(dataset)


def main() -> None:
    output_path = Path(os.getenv("OUTPUT_FILE", "results/responses.jsonl")).resolve()
    gguf_root = get_gguf_root()
    preferred_file = os.getenv("HF_MODEL_FILE", get_default_gguf_name())
    dataset_limit = int(os.getenv("DATASET_LIMIT", "5"))
    only_ids = {
        entry.strip()
        for entry in os.getenv("DATASET_IDS", "").split(",")
        if entry.strip()
    }

    model_path = pick_gguf_file(gguf_root, preferred_file)
    llm = build_llm(model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_tokens = int(os.getenv("MAX_TOKENS", "128"))
    temperature = float(os.getenv("TEMPERATURE", "0.2"))

    with output_path.open("w", encoding="utf-8") as handle:
        for entry in DATASETS:
            dataset_id = entry["id"]
            if only_ids and dataset_id not in only_ids:
                continue

            records = load_dataset_records(entry)
            if dataset_limit > 0:
                records = records[:dataset_limit]

            for record in records:
                prompt, ground_truth = record_to_prompt(dataset_id, record)
                if not prompt:
                    continue
                response = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["</s>"],
                )
                text = response["choices"][0]["text"].strip()
                payload = {
                    "dataset_id": dataset_id,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "response": text,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model_path": str(model_path),
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    print(f"Saved responses to {output_path}")


if __name__ == "__main__":
    main()
