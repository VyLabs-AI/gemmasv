# Hardware, runtime, and numerical boundaries

## Reference system

The training-free 1B/4B graft and behavioral path run on an Apple M3 Ultra
using PyTorch/MPS; the solver seed and bandwidth sample are deterministic.
Certificate runs use a separate float64 CPU model. Legacy recovery and 12B
diagnostics require substantially more compute and memory.

Record the following with every rerun:

- operating system and architecture;
- CPU/GPU model and available memory;
- Python, NumPy, SciPy, CVXPY, Torch, Transformers, PEFT, and MLX versions;
- model revision, adapter hash, command, seed, and precision;
- warm-up count and repetitions for timings.

Do not treat smoke-run latency as a paper result.

## Runtime envelope

- `run_quick.sh`: seconds to a few minutes; CPU; no model download.
- `run_headline.sh --quick`: minutes after dependencies/model code are cached;
  random-model wiring only.
- Training-free 1B/4B admission: seconds per serialized request after model
  load; no optimizer or adapter is involved.
- 1B float64 certificate: tens of seconds to minutes per target on the reference
  CPU, depending on memory length and decrement fallbacks.
- 1B three-field whole-record certificate: four output probes take roughly
  4.3--6.9 minutes per record on the reference CPU. The coupled block decrement
  passes post-verification on 86--96% of affected head-gates in the evaluated
  records; the remainder route to exact fixed-$C$ refit, so runtime grows with
  the fallback count.
- full attack, relearning, and LiRA sweeps: hours.
- probabilistic `leak@k`/TOFU-Pair sampling: the smoke path is roughly a minute
  on the M3 Ultra. With cached prefill-and-decode (the default; grafted layers
  reuse chunk-boundary gate solves), a full 200-sample shard takes roughly
  3-4 minutes and the 200-shard paper grid completes in well under a day; the
  `--no-kv-cache` full-recompute path is ~12x slower. MPS batches above 16 are
  intentionally rejected. Run one orchestrator at a time: two concurrent
  MPS workers measured only ~1.28x aggregate throughput, and a third loses it.
- Legacy 4B/12B recovery or paper-scale reevaluation: accelerator workloads; plan in
  hours to days and record peak memory.
- Legacy three-seed 4B run on the 512-GB M3 Ultra: two seed pipelines in
  parallel completed cleanly; a three-way production-batch smoke left one MPS
  worker hung during teardown and was rejected. Parallel recovery stage 2 took
  about 998 minutes per seed versus 554 minutes for the later single worker;
  the full three-recovery/three-control orchestration took 58.4 hours.

These are scheduling ranges, not performance claims.

## Deterministic and non-deterministic components

The strongest deterministic checks are the CPU float64 solver/refit comparisons
with explicit tolerances. Fixed seeds make data order and target selection
repeatable.

The following may vary while remaining valid:

- MPS/CUDA reduction order and greedy decoding near probability ties;
- convex-solver choices for ill-conditioned or non-unique optima;
- decrement fallback count on a new model/library stack;
- wall-clock timing;
- first-use model and dataset downloads;
- PDF bytes across TeX engines.

For that reason:

- judge deletion using the measured KL and raw probabilities, not generated text
  alone;
- report decrement fallbacks rather than hiding them;
- report median, tail, and worst-case certificate values;
- compare against a retained-key refit and separately against a fully repacked,
  never-ingested context.

## Precision scope

“Exact” names the algebraic decrement/refit equivalence of the support-vector
memory. Realized floating-point agreement is measured, not assumed:

- the training-free all-record sweeps reach maximum output KL `1.25e-10` at
  1B and `6.48e-11` at 4B over 32 probes per scale;
- reverse decrement falls back to exact refit in 39% (1B) and 47% (4B) of
  head-gates, preserving equality while reducing reverse-path coverage;
- the superseded recovered-model diagnostics include heavier conditioning
  tails at 4B;
- the legacy recovered-readout 12B median is around `1e-9`, with larger worst
  cases; the predeclared training-free 12B extension has no certificate;
- multi-token sequential deletion can accumulate numerical error.

The paper and demo must show the measured value and avoid rounding all of these
regimes to “zero.”

