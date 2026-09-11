# Third-party notices

The root Apache-2.0 license applies to the original replication code and
release tooling in this snapshot. It does not replace upstream licenses.

## Figure tooling

`figure_tools/` is distributed under the MIT license in
`figure_tools/LICENSE`. Its npm dependencies retain their own licenses as
recorded in `package-lock.json`.

The renderer installs Noto Sans through `@fontsource/noto-sans`; Noto fonts are
licensed under the SIL Open Font License 1.1. The generated PDFs embed subsets
of that font.

## TeX and bibliography support

- `paper/fancyhdr.sty` is distributed under the LaTeX Project Public License
  (LPPL), version 1 or later.
- `paper/natbib.sty` and `paper/iclr2027_conference.bst` are distributed under
  the LPPL terms stated in their file headers.
- `paper/iclr2027_conference.sty` is the official ICLR formatting file,
  adapted historically from NIPS macros; its header and attribution are
  preserved verbatim.

## Model and dataset dependencies

Gemma, Kimi Linear, Qwen3.5, FineWeb-Edu, WikiText, TOFU, LongMemEval,
LongMemEval-V2, RULER, MIMIC, and other external models or datasets are not
redistributed. Reproducers must obtain them independently under their upstream
licenses and access terms.

The Qwen replay audit uses the Apache-2.0
`Qwen/Qwen3.5-4B-Base` checkpoint at pinned revision
`57370f0ea82c3cca33558a95212e032c344e5fd5`. Model weights are not
redistributed.

The optional demo-only natural-language resolver uses Google's `google-genai`
Python SDK, distributed under Apache-2.0, to access the Gemini API. The SDK,
Gemini service, credentials, and model weights are not redistributed. The
recorded resolver fixture requires neither the SDK nor network access.

The released LongMemEval compact16 aggregate contains no conversation text,
generated text, or full-vocabulary vectors. The phone replay includes only
hash-bound excerpts from the public benchmark and labels their benchmark
assembly. The separate decoded-response audit is source-bearing, local-only,
and not included.
