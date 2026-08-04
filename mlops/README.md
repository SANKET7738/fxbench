# mlops — Inference & Training for Formula Completion

Standalone PyTorch utilities for running batch **inference** and **SFT/LoRA
training** of language models for spreadsheet formula completion.

This package runs entirely locally: models are loaded directly from a Hugging
Face hub id or a local directory, and all outputs are written to the local
filesystem.

## Installation

```bash
pip install -e ./mlops
# With DeepSpeed (multi-GPU ZeRO / tensor parallelism):
pip install -e "./mlops[deepspeed]"
```

## Batch inference

```bash
python -m mlops.predict \
  --model Qwen/Qwen3-8B \
  --prompts prompts.jsonl \
  --output results.jsonl \
  --batch-size 8
```

`--model` is a Hugging Face hub id or a local path and is loaded as-is. Useful
flags:

| Flag | Purpose |
| --- | --- |
| `--prefix` | Force a string at the start of every generation (e.g. `'{"formula": "'`). |
| `--adapter-path` | Apply a PEFT/LoRA adapter on top of `--model` (the base). |
| `--base-model` | Informational base-model id, used only with `--adapter-path`. |
| `--trust-remote-code` | Allow models that ship custom modelling code. |
| `--inference-kwargs-base64` | Base64-encoded JSON of generation kwargs (temperature, top_p, num_return_sequences, …). |

### Prompt file format

`--prompts` is a JSONL file; each line may be any of:

```jsonc
{"id": "1", "prompt": "plain string prompt"}
{"id": "2", "prompt": "[{\"role\": \"user\", \"content\": \"...\"}]"}  // JSON string of messages
{"id": "3", "messages": [{"role": "user", "content": "..."}]}          // messages array
```

Results are written to `--output` (JSONL) and a sibling `.json` file in
`InferenceResults` format.

## Training (SFT / LoRA)

```bash
python -m mlops.trainer \
  --model Qwen/Qwen3-8B \
  --datapath training-data.jsonl \
  --epochs 1
```

LoRA is enabled by default; pass full-parameter SFT and DeepSpeed options as
needed, for example:

```bash
torchrun --nproc_per_node=2 -m mlops.trainer \
  --model Qwen/Qwen3-8B \
  --datapath training-data.jsonl \
  --use_deepspeed --deepspeed_stage 3
```

Checkpoints, the merged model, and training artifacts are saved under
`./outputs/`. External experiment trackers are disabled by default
(`TrainerConfig.report_to = []`); set it to e.g. `["wandb"]` or
`["tensorboard"]` to enable one.

## Package structure

```
mlops/
├── src/mlops/
│   ├── infer.py               # Inference engine (multi-GPU, DeepSpeed)
│   ├── predict.py             # Batch inference CLI
│   ├── trainer.py             # SFT/LoRA training (SFTTrainer + DeepSpeed)
│   ├── training_artifacts.py  # Persist run config, metrics, model card
│   ├── token_batching.py      # Token-budget batch sampler
│   └── deepspeed_config.json  # Default DeepSpeed ZeRO config
└── pyproject.toml
```

## Multi-GPU (single-node)

Multi-GPU runs on a **single machine** are supported for both inference and
training:

- **Inference** — DeepSpeed tensor parallelism or `device_map="balanced"`.
- **Training** — PyTorch DDP or DeepSpeed ZeRO (stages 0–3).

All GPUs must be on the same node; multi-node distributed execution is not
supported.
