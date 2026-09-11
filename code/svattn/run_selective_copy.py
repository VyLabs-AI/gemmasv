"""Go/no-go: selective retrieval under increasing redundant filler.

Run:
    PYTHONPATH=. \
        ./ferg/.venv/bin/python -m svattn.run_selective_copy
"""
import torch

from svattn.selective_copy import (SVAttnModel, RBFUniformModel, SoftmaxModel,
                                   LinearAttnModel, DeltaNetModel, train, evaluate,
                                   sv_state_size)

D_IN, DV_IN, N_DATA = 6, 3, 8
CFG = dict(steps=300, seqs=3, lr=0.03)


def main():
    torch.manual_seed(0)
    print(f"Selective retrieval: {N_DATA} data tokens + F filler; chance = {1/N_DATA:.3f}\n")
    print(f"{'filler':>7}{'L':>6}{'SV-Attn':>10}{'softmax':>10}{'DeltaNet':>10}{'linear':>10}{'SV state':>12}")
    for F in (8, 24, 56, 120):
        L = N_DATA + F
        a = {}
        a["sv"] = evaluate(train(SVAttnModel(D_IN, DV_IN, C=0.25, kpar=1.5), N_DATA, F, D_IN, DV_IN, **CFG), N_DATA, F, D_IN, DV_IN)
        a["sm"] = evaluate(train(SoftmaxModel(D_IN, DV_IN), N_DATA, F, D_IN, DV_IN, **CFG), N_DATA, F, D_IN, DV_IN)
        a["delta"] = evaluate(train(DeltaNetModel(D_IN, DV_IN), N_DATA, F, D_IN, DV_IN, **CFG), N_DATA, F, D_IN, DV_IN)
        a["lin"] = evaluate(train(LinearAttnModel(D_IN, DV_IN), N_DATA, F, D_IN, DV_IN, **CFG), N_DATA, F, D_IN, DV_IN)
        st = sv_state_size(SVAttnModel(D_IN, DV_IN, C=0.25, kpar=1.5), N_DATA, F, D_IN, DV_IN)
        print(f"{F:>7}{L:>6}{a['sv']:>10.3f}{a['sm']:>10.3f}{a['delta']:>10.3f}{a['lin']:>10.3f}"
              f"{st:>8.1f}/{L}", flush=True)

    print("\nDeltaNet is the strong delta-rule recall baseline (fixed-size state). "
          "Watch whether its fixed state holds the rare items as filler grows.")


if __name__ == "__main__":
    main()
