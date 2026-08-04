# FxBench

FxBench is a benchmark for evaluating large language models in the inline formula completion setting.

## Setup

Requires Python 3.12. FxBench uses `xlwings` for evaluation, which requires Windows with Excel available.

```powershell
uv sync
uv run python -m fxbench.cli.setup_fxbench
```

For prediction runs, copy `.env.example` to `.env` and fill in your own Azure OpenAI values. The local `.env` file is ignored by git.


## Build prompts

```powershell
uv run python -m fxbench.cli.build_prompts --benchmark FxBench --context-builder minisheet 
```

## Run predictions

```powershell
uv run python -m fxbench.cli.get_predictions --benchmark FxBench --context-builder minisheet --model your-model-alias --workers 1
```

## Ingest external predictions

```powershell
uv run python -m fxbench.cli.ingest --benchmark FxBench --context-builder minisheet --model your-model-alias --path path\to\inference-results.jsonl --force
```


## Evaluate

```powershell
uv run python -m fxbench.cli.evaluate --benchmark FxBench --context-builder minisheet --model your-model-alias --force
```


Evaluation executes formulas through Excel, so it requires a Windows environment with Excel available.


## Scripts

Standalone utility scripts (generating intent-cell annotations, the context coverage comparison) are documented separately in [`src/fxbench/scripts/README.md`](src/fxbench/scripts/README.md).

