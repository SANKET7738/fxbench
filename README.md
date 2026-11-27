# FxBench

FxBench is a benchmark for evaluating large language models in the inline
spreadsheet formula-completion setting. It contains 503 examples with target
cells, ground-truth formulas, and sanitized workbook snapshots.

- Dataset: [Hugging Face](https://huggingface.co/datasets/sanket-mhatre/fxbench)
- Paper: [*fxBench: Evaluating and Understanding Formula Suggestions in Spreadsheets*](https://arxiv.org/abs/2610.24729) ([PDF](https://arxiv.org/pdf/2610.24729))
- 🎉 NeurIPS 2026: [Accepted paper and poster](https://neurips.cc/virtual/2026/poster/139181)
- Code: [github.com/SANKET7738/fxbench](https://github.com/SANKET7738/fxbench)

## Setup

FxBench requires Python 3.12. The commands below use
[`uv`](https://docs.astral.sh/uv/):

```powershell
uv sync
uv run python -m fxbench.cli.setup_fxbench
```

The setup command downloads the public dataset from Hugging Face and
materializes the workbook snapshots in `Assets/workbooks/`.

For prediction runs, copy `.env.example` to `.env` and enter your own Azure
OpenAI configuration. The local `.env` file is ignored by Git.

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

Evaluation executes formulas through Excel using `xlwings`, so evaluation
requires Windows with Microsoft Excel installed.

## Updated results (November 2026)

Newer models evaluated on all 503 FxBench examples. The prompts, harness and evaluator are the
same as in the paper, with a single run per model. ACC is abstain-aware accuracy (%). The paper's
models are shown for reference. Per-example predictions and evaluations are in
`Assets/Evaluations/FxBench/content_baselines/` and `Assets/Evaluations/FxBench/oracle_analysis/`.

| Representation | GPT-5.2 (paper) | Sonnet 4.6 (paper) | GPT-6.2 Astra | Sonnet 5.5 | Gemini 4 Neon | GPT-6.2 Sol | Opus 5.5 | Fable 5.2 | Gemini 4 Argon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HTML | 58.3 | 60.0 | 72.6 | 66.8 | 69.4 | 70.0 | 69.4 | 70.0 | 63.4 |
| CSV | 49.7 | 53.3 | 73.0 | 60.4 | 68.6 | 67.8 | 67.4 | 67.8 | 63.6 |
| JSON | 56.9 | 60.4 | 72.2 | 65.6 | 69.2 | 70.4 | 68.4 | 69.4 | 62.4 |
| Markdown | 53.3 | 54.7 | 73.6 | 65.4 | 68.2 | 69.6 | 67.8 | 68.8 | 61.2 |
| Pipe-delimited | 58.8 | 59.8 | 73.4 | 65.8 | 69.0 | 69.8 | 67.6 | 68.2 | 62.8 |
| Viewport | 56.9 | 58.4 | 68.8 | 61.6 | 64.4 | 66.2 | 64.4 | 65.8 | 58.3 |
| SpreadsheetCoder | 52.9 | 57.1 | 67.8 | 57.7 | 61.2 | 63.8 | 62.0 | 64.4 | 54.5 |
| SheetCompressor | 54.7 | 50.7 | 70.6 | 62.6 | 67.8 | 67.0 | 65.8 | 66.8 | 58.1 |
| Minisheet | 42.3 | 49.7 | 67.4 | 55.7 | 61.6 | 64.4 | 64.4 | 63.0 | 55.7 |

Oracle ladder (ACC, %):

| Oracle variant | GPT-5.2 (paper) | GPT-6.2 Astra |
|---|---:|---:|
| Base oracle | 66.2 | 78.9 |
| All signals | 76.9 | 89.9 |
| All signals + NL | 97.2 | 98.8 |

Run settings, failure counts and parity checks are in [`RUNS.md`](RUNS.md).

## Repository structure

- `src/fxbench/`: benchmark loading, prompt construction, inference, and evaluation code.
- `Assets/Annotations/`: released benchmark annotations.
- `Assets/Evaluations/`: released evaluation outputs and model predictions.
- `Assets/SampledPredictions/`: sampled outputs used for analysis.
- `src/fxbench/scripts/`: standalone analysis and data-preparation utilities.

Standalone scripts for intent-cell annotations and context-coverage analysis
are documented in [`src/fxbench/scripts/README.md`](src/fxbench/scripts/README.md).

## Dataset provenance and responsible use

FxBench is derived from
[Sheetpedia](https://huggingface.co/datasets/tianzl66/Sheetpedia_xlsx). Workbook
metadata is sanitized before release, including document properties,
classification-label metadata, custom properties, comment/person metadata, and
local path markers. The benchmark is intended for research on spreadsheet
formula suggestion and evaluation; it does not guarantee that generated
formulas are correct or safe for production spreadsheets.

## Licenses

Source code is released under the [MIT License](LICENSE). The FxBench dataset
and benchmark data artifacts are released under
[CC BY-SA 4.0](DATA_LICENSE.md), including attribution to Sheetpedia. Please
review both files before redistributing code and data together.

## Citation

If you use FxBench, please cite the [paper](https://arxiv.org/abs/2610.24729):

```bibtex
@misc{mhatre2026fxbench,
  title = {{fxBench}: Evaluating and Understanding Formula Suggestions in Spreadsheets},
  author = {Sanket Mhatre and Sumit Gulwani and Vu Le and Yasharth Bajpai and Gust Verbruggen},
  year = {2026},
  eprint = {2610.24729},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2610.24729}
}
```
