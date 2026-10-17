# FxBench

FxBench is a benchmark for evaluating large language models in the inline
spreadsheet formula-completion setting. It contains 503 examples with target
cells, ground-truth formulas, and sanitized workbook snapshots.

- Dataset: [Hugging Face](https://huggingface.co/datasets/sanket-mhatre/fxbench)
- Paper: *fxBench: Evaluating and Understanding Formula Suggestions in Spreadsheets*
- Code: [github.com/SANKET7738/fxbench](https://github.com/SANKET7738/fxbench)

The arXiv link and identifier will be added after the preprint is published.

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

## Citation

If you use FxBench, please cite the accompanying paper. The arXiv identifier
will be added after publication.

```bibtex
@article{mhatre2026fxbench,
  title   = {fxBench: Evaluating and Understanding Formula Suggestions in Spreadsheets},
  author  = {Mhatre, Sanket and Gulwani, Sumit and Le, Vu and Bajpai, Yasharth and Verbruggen, Gust},
  year    = {2026}
}
```

The same citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Licenses

Source code is released under the [MIT License](LICENSE). The FxBench dataset
and benchmark data artifacts are released under
[CC BY-SA 4.0](DATA_LICENSE.md), including attribution to Sheetpedia. Please
review both files before redistributing code and data together.
