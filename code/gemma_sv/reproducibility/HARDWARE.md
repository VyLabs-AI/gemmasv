# Hardware and numerical boundaries

The training-free Gemma and behavioral studies used an Apple M3 Ultra with
PyTorch/MPS and MLX. Certificate checks use a separate CPU float64 model.
The [reference environment](ENVIRONMENT.md) and study protocols record versions.

Record the operating system, accelerator and memory, Python/package versions,
model revision, dtype, command, seed and timing repetitions with each rerun.

## Compute requirements

- `run_quick.sh`: CPU checks, no model download.
- `python -m gemma_sv.smoke_test`: random-model wiring with full dependencies.
- Admission and paired quality: pretrained model weights and an accelerator;
  no optimizer or adapter is involved in the paper's training-free configuration.
- Float64 certificates: enough CPU memory for a separate model; runtime depends
  on memory length and decrement fallbacks.
- Sampling, elicitation, relearning and LiRA sweeps: extended model workloads.
  The recorded runner restricts MPS batch sizes above 16.

Scheduling estimates or smoke-run timings are not paper performance results.
The exact measured costs, repetition counts and scope are in the protocols and
results under `../../../journal_evidence/`.

## Numerical checks

CPU float64 solver/refit comparisons use explicit tolerances. Fixed seeds make
selection and ordering repeatable, but accelerator reduction order, decoding
near probability ties, solver choices, fallback counts and wall-clock timings
can vary with hardware and packages.

The all-record sweeps reach maximum output KL of `1.25e-10` at 1B and
`6.48e-11` at 4B over 32 probes per scale. Reverse decrement routes to refit in
`396/640` (61.9%) and `2,020/3,040` (66.4%) affected solve attempts, respectively.
The denominator includes affected solves, not every enumerated gate. Refitting
completes those updates; the measured training-free 12B extension has no
certificate result.

Report the measured KL, probabilities and fallbacks. Keep agreement with a
retained-key refit separate from agreement with a fully rebuilt, never-ingested
context. Floating-point agreement is measured, not assumed from the algebra.
