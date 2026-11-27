# Run details: updated results (November 2026)

These are the runs behind the "Updated results" tables in the [README](README.md). Every run used
the paper's frozen prompts (503 examples per representation) and its prediction harness, response
parser and Excel-based evaluator, all unchanged. ACC is the paper's abstain-aware accuracy: a
correct formula or a correct abstention counts as a hit, and errors stay in the denominator. Each
model ran once.

## Settings

The paper used two kinds of setup. GPT-5.2 was run as a reasoning model at the provider's default
effort, which at the time produced light reasoning (today that corresponds to `low`; see Parity).
Sonnet 4.6 and Qwen3-8B were run as non-reasoning models. Each new model is matched to whichever
of those two setups it fits:

| Model | Matched to | Max output tokens | Temperature | Reasoning / thinking | Run date |
|---|---|---:|---|---|---|
| GPT-6.2 Astra | GPT-5.2 (reasoning) | 16,384 | not sent | reasoning effort `low` | 2026-11-25 |
| GPT-6.2 Sol | GPT-5.2 (reasoning) | 16,384 | not sent | reasoning effort `low` | 2026-11-25 (pipe-delimited, Viewport, Minisheet); 2026-11-27 (others) |
| Gemini 4 Neon | GPT-5.2 (reasoning) | 16,384 | not sent | reasoning effort `low` | 2026-11-27 |
| Gemini 4 Argon | GPT-5.2 (reasoning) | 16,384 | not sent | reasoning effort `low` | 2026-11-27 |
| Sonnet 5.5 | Sonnet 4.6 (non-reasoning) | 4,000 | not sent (rejected by the model) | thinking off | 2026-11-27 |
| Opus 5.5 | GPT-5.2 (reasoning) | 16,384 | not sent (rejected by the model) | thinking cannot be turned off; effort `low` | 2026-11-27 |
| Fable 5.2 | GPT-5.2 (reasoning) | 16,384 | not sent (rejected by the model) | thinking cannot be turned off; effort `low` | 2026-11-27 |

Notes:
- Low effort is the setting that reproduces the paper's GPT-5.2 behaviour (see Parity below).
- Gemini 4 Neon and Argon, Sonnet 5.5, Opus 5.5 and Fable 5.2 were first run at provider defaults:
  sampling on for Gemini, and thinking on by default for the Claude models. Those first runs were
  replaced by the paper-matched runs above. On default thinking, Gemini 4 Neon used its whole
  output limit on 152 examples and returned no answer.

## Parity

Before the new models were run, the paper's own models were re-run through the same code path:

| Check | Paper | Re-run |
|---|---:|---:|
| GPT-5.2, pipe-delimited, reasoning effort `low` | 58.8 | 59.4 |
| GPT-5.2, pipe-delimited, provider-default effort | 58.8 | 47.1 |
| Sonnet 4.6, JSON, non-reasoning | 60.4 | 57.9 |

- At default effort, today's GPT-5.2 barely reasons: median output is 20 tokens, against the
  paper's 157. At low effort the median is 175 and accuracy matches.
- Sonnet 4.6 is 2.5 points below the paper. Its runs sample at temperature 0.7, and one standard
  error at n = 503 is about 2.2 points.

## Failed calls

Failed calls are scored the way the harness scores them: as a missing formula. That is an
abstention, so it counts as correct only where the reference is an abstention.

**Prompt longer than the model's context window.** These calls cannot succeed without changing
the prompt, so they were not retried:

| Model | Cases |
|---|---|
| GPT-6.2 Astra | fxbench-94 (HTML, JSON, Markdown, pipe-delimited) |
| GPT-6.2 Sol | fxbench-94 (HTML, JSON, Markdown, pipe-delimited) |
| Sonnet 5.5 | fxbench-94 (CSV, HTML, JSON, Markdown, pipe-delimited), fxbench-368 (HTML) |
| Opus 5.5 | fxbench-94 (CSV, HTML, JSON, Markdown, pipe-delimited), fxbench-368 (HTML) |
| Fable 5.2 | fxbench-94 (CSV, HTML, JSON, Markdown, pipe-delimited), fxbench-368 (HTML) |
| Gemini 4 Neon | fxbench-94 (HTML, JSON, Markdown, pipe-delimited), fxbench-368 (HTML) |
| Gemini 4 Argon | fxbench-94 (HTML, JSON, Markdown, pipe-delimited), fxbench-368 (HTML) |

**No answer returned.** For fxbench-489 (Gemini 4 Argon, JSON), the call failed three times
because the provider's reply carried no usage information. It is scored as a missing answer.
Argon answered this example in every other representation.

None of the scored runs contains an empty reply caused by the output-token limit.

**Evaluator errors.** Excel's automation interface fails to evaluate two predicted formulas. These
failures repeated on every re-evaluation, so they stay as `Error`:
- fxbench-418 (Sonnet 5.5, Markdown);
- fxbench-56 (Sonnet 5.5, SpreadsheetCoder).

Every other `Error` outcome seen during these runs was either transient (re-evaluating the example
gave TP or FP) or in a first run that has since been replaced.

## Replies with prose before the JSON

The prompt asks for JSON only. Some models write an explanation first:

| Model | Prose before the JSON |
|---|---:|
| Sonnet 5.5 | 3,230 / 4,521 replies |
| Opus 5.5 | 4,190 / 4,521 replies |
| Fable 5.2 | 224 / 4,521 replies |
| GPT-6.2 and Gemini models | 0 |

This prose is visible answer text, not reasoning output. Sonnet 5.5 ran with thinking off and still
writes it, so it reflects the model not following the JSON-only instruction.

The parser takes the final JSON object in each reply. A hand check found that it recovered the
formula from every prose reply except three, where the model's own output is not valid JSON:

- **fxbench-465 (Sonnet 5.5, Minisheet).** The quote escaping inside the formula is inconsistent,
  so no formula is extracted. The intended formula essentially matches the reference.
- **fxbench-413 (Sonnet 5.5, Viewport).** The reply ends with a bare formula and no JSON object.
- **fxbench-360 (Opus 5.5, Minisheet).** An unescaped `""` truncates the extracted formula. The
  reference for this example is an abstention, so the result is FP either way.

The parser was not changed. These three examples are scored as the harness scored them.
