# fxbench scripts

Standalone, reproducible utility scripts. Run them as modules from the repo root
(`uv run python -m fxbench.scripts.<name>`). These are separate from the `fxbench.cli`
commands documented in the top-level [README](../../../README.md).

## Generate intent-cell annotations

`intent_cells_annotator.py` — produces per-problem intent annotations (`direct_intent`,
`data_description`, `data_range`, and a short `intent` string) using an LLM-as-judge over each
ground-truth formula and its surrounding cells. Problems are loaded from the published FxBench
benchmark.

```powershell
uv run python -m fxbench.scripts.intent_cells_annotator --workers 8 --model your-model-alias
```

Output is written to `Assets\Annotations\FxBench\intent_cells.jsonl` (one JSON object per line,
with resume support).

Flags:

- `--dry-run --take 2` — print the prompts for a few problems without calling the LLM
- `--take N` — limit to the first N problems
- `--workers N` — parallel LLM calls
- `--model <alias>` — model alias/deployment (resolved via `MODEL_ALIASES`; falls back to `DEPLOYMENT_NAME`)
- `--force` — overwrite existing annotations
- `--output PATH` — change the destination
- `--split {train,test}` — benchmark split
- `--width N` — surrounding-cell width (default 4)

Generating annotations requires Azure OpenAI access configured in `.env`.

## Context coverage comparison

`context_coverage_comparison.py` — for each context strategy, reports the percentage of FxBench
problems where the cells a model needs (formula references and intent cells) are exposed by that
strategy's context, plus the full-prompt token cost (cl100k). Coverage is computed from the intent
annotations in `Assets\Annotations\FxBench\intent_cells.jsonl` and the materialised workbooks.

```powershell
uv run python -m fxbench.scripts.context_coverage_comparison --workers 8
```

Prints a formatted table over PipeDelimited, Viewport (±20/±30/±40), SpreadsheetCoder,
SheetCompressor, and MiniSheet (w=1–4).

Flags:

- `--take N` — limit to the first N problems (quick smoke test)
- `--workers N` — parallel workers for token rendering
- `--split {train,test}` — benchmark split
- `--upper {v2,v3,v4}` — Upper-limit definition (default v2)
- `--intent-file PATH` — use an alternate intent_cells JSONL (default `Assets\Annotations\FxBench\intent_cells.jsonl`)

A full run is heavy (coverage passes plus token rendering over the whole benchmark); use `--take`
for iteration.

### Intent annotation sets

Two annotation sets ship under `Assets\Annotations\FxBench\`:

- `intent_cells.jsonl` — generated with gpt-5.4 (the default used by the comparison).
- `intent_cells_gpt-5.2.jsonl` — the paper's original gpt-5.2 annotations, re-keyed to `fxbench-N`
  ids (379 mapped, the rest `null` for ABSTAIN or unmatched problems). Provided for reference and
  reproducibility of the paper's tables.

Run the comparison against the paper's set with:

```powershell
uv run python -m fxbench.scripts.context_coverage_comparison --intent-file Assets\Annotations\FxBench\intent_cells_gpt-5.2.jsonl
```

## Content baseline comparison

`content_baseline_comparison.py` — prints the content-strategy baseline table (the paper's
Table 3). For each context strategy and model it reports the abstain-aware accuracy
(**ACC** = Pass@1, `(TP+TN)/N`), the upper-normalized accuracy (**ACC†** = ACC ÷ the strategy's
context-evidence upper limit), and the mean expected-reference range overlap (**OVL**), grouped
into Full-sheet (F), Windowed (W), and Compact (D) strategies across GPT-5.2, Sonnet-4-6, and
Qwen3-8B.

```powershell
uv run python -m fxbench.scripts.content_baseline_comparison
```

It is pure aggregation over committed per-record evaluation files — no Excel, workbooks, or
benchmark loading required.

### Evaluation data

The table is read from per-record evaluation files under
`Assets\Evaluations\FxBench\content_baselines\<Builder>\<Model>\evaluation.jsonl` (9 strategies ×
GPT-5.2 / Sonnet-4-6 / Qwen3-8B = 27 files). Each line records one FxBench problem:

- `id` — `fxbench-N`
- `outcome` — abstain-aware `TP` / `FP` / `TN` / `FN` / `Error`
- `range_overlap` — expected-reference overlap in `[0, 1]` (or `null` when not applicable)
- `formula` — `{predicted, expected}`
- `metrics` — the raw per-record evaluation metrics (`ValueShown`, `Syntax`, `Edit`, …)

These files are committed as the reference model outputs for analysis, so the table reproduces
without re-running inference or Excel evaluation.

## Oracle analysis

`oracle_analysis.py` — prints the oracle-results table (the paper's Table 4). For each additive
Oracle variant it reports the strict non-abstain accuracy (**TP** = `TP/N`, excluding
abstention), the execution accuracy (**ACC** = `(TP+TN)/N`), and the **Δ** improvement in ACC over
the base oracle, at GPT-5.2. Variants are additive over `base` and ordered by the scope of the
revealed signal (target-local → reference-local → sheet-wide → stacked → out-of-sheet).

```powershell
uv run python -m fxbench.scripts.oracle_analysis
```

It is pure aggregation over committed per-record evaluation files — no Excel, workbooks, or
benchmark loading required.

### Evaluation data

The table is read from per-record evaluation files under
`Assets\Evaluations\FxBench\oracle_analysis\<Builder>\gpt-5.2\evaluation.jsonl` (the 8 Oracle
variants). Each line records one FxBench problem:

- `id` — `fxbench-N`
- `outcome` — abstain-aware `TP` / `FP` / `TN` / `FN` / `Error`
- `formula` — `{predicted, expected}`
- `metrics` — the raw per-record evaluation metrics (`ValueShown`, `Syntax`, `Edit`, …)

These files are committed as the reference oracle model outputs for analysis, so the table
reproduces without re-running inference or Excel evaluation.

## Fine-tuning results

`finetuning_results.py` — prints the combined fine-tuning results table (the paper's Table 5):
sampled **pass@1 / pass@3 / Best@8** (eight completions per problem) for the model-size comparison on
the Curated dataset and the dataset-choice comparison at fixed Qwen3-8B scale, with base-context and a
GPT-5.2 closed-model reference.

```powershell
uv run python -m fxbench.scripts.finetuning_results
```

It is pure aggregation over the committed pass@k file — no Excel, workbooks, or benchmark loading.

### Data

- `Assets\Evaluations\FxBench\finetuning_results.json` — the aggregate pass@k per row (pass@1, pass@3,
  best@8 / total, training-record count) that the table is printed from.
- `Assets\SampledPredictions\FxBench\<slug>\sampled.json` — the eight sampled completions per problem
  for each of the 12 models, committed as the analysis reference. Each `results` entry holds
  `{id: fxbench-N, predictions: [8 completions]}`; `parameters` carry the paper-facing model label,
  training-data name, and record count.

Per-completion correctness was not re-evaluated, so the table uses the saved aggregate pass@k.

## Failure summary

`failure_summary.py` — prints a text breakdown of *why* a run''s failures failed, for a single
evaluated file (no table, no plot). Failure rows (`outcome` not in {TP, TN}) are bucketed into six
categories — Wrong formula type / Wrong args / Wrong range / Partial correct / Execution error /
Wrong abstain — with counts and percentages.

```powershell
uv run python -m fxbench.scripts.failure_summary Assets\Evaluations\FxBench\oracle_analysis\OracleKitchenSinkContextBuilder\gpt-5.2\evaluation.jsonl
```

It reads one evaluated per-record file (the `evaluation.jsonl` schema — failure *types* require
outcomes). The buckets follow the paper''s deterministic failure taxonomy but are derived offline from
the committed per-record metrics plus a light formula comparison, so the counts approximate (rather
than exactly reproduce) the paper''s prose-AST labels.

### Raw predictions

Beside every `evaluation.jsonl` under `content_baselines/` and `oracle_analysis/`, the raw model
output is committed as `predictions.jsonl` (`{id: fxbench-N, raw_response, answer}`) as a reference
for analysis.

## benchmark_curation/

Internal benchmark-curation pipeline modules (dataset filtering, sketch/signature dedup,
programmatic and LLM annotations, ranking/selection, and snapshot generation) used to build
FxBench from raw data. These are not part of the standard prediction/evaluation workflow.
