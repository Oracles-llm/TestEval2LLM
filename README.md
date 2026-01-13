Smoke-test repo for running a Hugging Face model with `llama-cpp-python`.

Default model:
- Edit `config.py` and update `DEFAULT_REPO_ID` to switch models in one place.

Prepare the model (download weights, then convert to GGUF):
```bash
python -m pip install -r requirements.txt
python download_model.py
python make_gguf.py
```

Run the model (interactive):
```bash
python run_model.py
```

Run prompt suite and save responses:
```bash
python run_prompts.py
```

Single prompt (non-interactive):
```bash
set PROMPT=Say hello in one sentence.
python run_model.py
```

Environment overrides:
```bash
set HF_REPO_ID=Qwen/Qwen2.5-3B-Instruct
set HF_MODEL_FILE=Qwen_Qwen2.5-3B-Instruct.f16.gguf
set MODEL_DIR=models
set GGUF_DIR=gguf
set N_GPU_LAYERS=0
python run_model.py
```

Notes:
- The `Qwen/Qwen2.5-3B-Instruct` repo provides safetensors, not GGUF. The model is downloaded into `models/Qwen_Qwen2.5-3B-Instruct`, then converted to `gguf/Qwen_Qwen2.5-3B-Instruct.f16.gguf` using the official `llama.cpp` conversion script.
- Set `HF_MODEL_FILE` to load a specific GGUF file from `GGUF_DIR`.
- Set `FORCE_CONVERT=1` to rebuild the GGUF after editing the downloaded model files.
- `run_prompts.py` reads `prompts.txt` and writes `results/responses.jsonl`.
