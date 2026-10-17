# Dataset and benchmark-artifact license

The FxBench source code is licensed separately under the MIT License in
[`LICENSE`](LICENSE).

The FxBench dataset, including workbook snapshots, formulas, labels, metadata,
annotations, and other benchmark data or derived data artifacts, is licensed
under the [Creative Commons Attribution-ShareAlike 4.0 International License
(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

Files under `Assets/` are benchmark artifacts or derived experimental data
unless a file states otherwise. They are not covered by the repository's MIT
software license.

FxBench is derived from
[Sheetpedia](https://huggingface.co/datasets/tianzl66/Sheetpedia_xlsx), which is
also distributed under CC BY-SA 4.0. FxBench selects and transforms workbook
content, defines formula-completion targets, sanitizes workbook metadata, and
adds benchmark annotations and evaluation artifacts. These changes do not
imply endorsement by the Sheetpedia authors.

When redistributing or adapting FxBench data, you must provide attribution,
link to CC BY-SA 4.0, indicate whether changes were made, and distribute
adaptations under the same or a compatible license. This notice is a practical
summary and does not replace the license text.

Suggested attribution:

> FxBench by Sanket Mhatre, Sumit Gulwani, Vu Le, Yasharth Bajpai, and Gust
> Verbruggen, derived from Sheetpedia, licensed under CC BY-SA 4.0.
