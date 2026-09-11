# Release validation

Checked 11 September 2026 from this staged snapshot, without model inference:

- The journal bundle verifier matched all 110 distributed payload files with zero unexpected files.
- The original three-file pytest command passed 34 tests and skipped 3 optional local-artifact checks, with no failures. The two added publication-test files passed all 12 tests, validating the exact decoded and suffix macros against their hash-bound inputs.
- Both documented journal numerical reanalyses exactly reproduced their saved scientific payloads. Only execution provenance and completion timestamps were excluded from the comparison.
- All released Python files parsed successfully.
- Original and independently rebuilt TeX archives produced identical extracted text and rendered pages. The 23-page journal PDF and all 19 pages of the public archive passed build checks; the journal pages were visually reviewed.
- The stage includes no weights, source conversation collections, token arrays, credential patterns or credentialed clinical data. Legacy synthetic demo fixtures remain explicitly synthetic.

The snapshot's cp_svm solver files and core Gemma gate modules are byte-identical to the canonical workspace. The three changed svattn solver modules differ only in explanatory docstrings. Optional generation checkpointing and the live assistant demo have evolved separately; this snapshot records the paper implementation rather than silently substituting those later tools. The journal study preserves its separate exact execution snapshot.

These checks verify packaging, saved-result analysis, and implemented contracts. They do not independently rerun the model experiments or supply independent human verification of scientific conclusions.

The completed decoded pipeline was then added with all 12 original implementation fingerprints matched exactly. Its 28 offline contract tests passed, checking the 96-history cohort, clustered analysis and mocked response runner; five cases requiring upstream source/model assets were excluded. The standalone test adaptations isolate GPU seeding for fake runtimes and replay the historical environment input; production checks are unchanged. See `code/REPRODUCE_DECODED.md`.
