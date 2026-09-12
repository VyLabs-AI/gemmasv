# Repository validation

Checked 11 September 2026. The README and reproduction guides now point to the
current paper's results. No model inference or new experiment was run.

- `journal_evidence/verify_bundle.py` matched all 110 payload files and found
  no unexpected files. All 112 study files, including the two manifests,
  remain byte-identical to the preceding release.
- `code/gemma_sv/reproducibility/run_quick.sh` passed 46 result/solver tests
  and 28 decoded-pipeline contract tests. Three tests requiring original local
  source artifacts were skipped; five cases requiring upstream sources or
  model/tokenizer assets were excluded.
- All 12 implementation files pinned by the decoded response-generation
  protocol retain their exact SHA-256 values.
- All 235 remaining Python files parse. Retained Python sources, result/protocol
  JSON and numerical fixtures are unchanged.
- Three recovery-only shell orchestrators and four recovery-training Python
  modules were removed after checking that no current pipeline or test imports
  or invokes them. The complete source-tree inventory in the frozen study
  provenance remains unchanged. Current result commands use the individual
  evaluation modules and the supported quick-check script.
- Shared recovery, retrieval and solver modules remain where current model
  and analysis entry points import them. Benchmark protocols and licenses
  remain intact; the repository contains no manuscript build package.

These checks establish package integrity and reconstruction from the supplied
measurements. Live model reruns require the model/data assets and execution
environment documented in the protocols. Accelerator checks were not run and
are not recorded as passing. Private response sets and review packets are not
distributed. The three data-only numerical fixtures under `code/tests/expected/`
require no TeX installation.
