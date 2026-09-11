# Release validation

Checked 11 September 2026 after trimming the repository to result reproduction.
No model inference or new experiment was run.

- The frozen study bundle verifier matched all 110 payload files with no
  unexpected files. Every frozen study file is byte-identical to the preceding
  release, including its two manifests and original execution snapshots.
- Result reconstruction and exact solver checks passed 46 tests. Three tests
  requiring original local source artifacts were skipped.
- The completed decoded-pipeline suite passed 28 offline tests. Five cases
  requiring original upstream sources or model/tokenizer assets were excluded.
- All 12 implementation files bound by the historical response-generation
  authorization retained their exact SHA-256 values.
- All retained Python sources parse. The three numerical fixtures were moved
  byte-for-byte from the former paper directories; only their test paths changed.
- Paper bundles, Node rendering dependencies, Kimi/Qwen utilities, independent
  SV experiments and hosted-demo UI/API were removed. Runtime dependencies used
  by Gemma experiments and existing licenses were retained.

`code/gemma_sv/reproducibility/run_quick.sh` runs the verified 46-test and
28-test commands. The earlier broader legacy suite was also attempted: the
optional MLX RNG test reported that this sandbox has no Metal device, and a
subsequent accelerator certificate test aborted during device initialization.
Those accelerator checks are not part of the supported offline command and
are not recorded as passing. Their historical production code is unchanged.

The preceding release exactly reproduced both saved study analyses; their
analyzers and measured inputs remain unchanged. These checks establish package
integrity and source-free result reconstruction. They do not independently
rerun model experiments, reproduce private review packets, or establish
scientific conclusions beyond the measured scope.

The repository preserves its existing Git history. The current manuscript is
cited through one arXiv record; manuscript compilation is outside release scope.
