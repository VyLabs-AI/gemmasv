# Third-party notices

The project Apache-2.0 license applies to original replication code and release
tooling. It does not replace upstream licenses.

## Model and dataset dependencies

Gemma, Kimi Linear, Qwen3.5, FineWeb-Edu, WikiText, TOFU, LongMemEval, MemOps,
UltraChat, RULER, MIMIC, and other external models or datasets are not
redistributed. Reproducers must obtain them independently under their upstream
licenses and access terms.

The longitudinal-conversation manifest is derived from MemOps commit
`312af65e2c7b6d1b70f062ffa8b4cde32aaf6f35`, released under MIT. MemOps uses
UltraChat carrier conversations. The manifest and aggregate reports store only
source identifiers, hashes, token geometry, and aggregate results.

The Qwen replay audit uses the Apache-2.0
`Qwen/Qwen3.5-4B-Base` checkpoint at pinned revision
`57370f0ea82c3cca33558a95212e032c344e5fd5`. Model weights are not
redistributed.

The optional demo-only natural-language resolver uses Google's `google-genai`
Python SDK, distributed under Apache-2.0, to access the Gemini API. The SDK,
Gemini service, credentials, and model weights are not redistributed. Use of
the hosted service remains subject to Google's applicable API terms and data
handling policies. A committed structured outcome over synthetic public demo
metadata may be redistributed only when it is labeled as recorded; replaying
that outcome requires neither the SDK nor network access.

## ICLR formatting support

The official ICLR style and bibliography files retain the license and
attribution notices embedded in their source headers. Bundled LaTeX support
files retain their own upstream terms.
