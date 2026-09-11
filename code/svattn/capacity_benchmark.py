"""Fair capacity benchmark: orthogonal-key recall as the number of distinct
items grows, with a fixed block of redundant filler.

Keys are near-orthogonal (so inner-product memories -- DeltaNet, softmax -- can
retrieve as well as the distance-kernel methods; this removes the
similarity-function confound). We then grow the number of DISTINCT items while
holding filler fixed, which stresses fixed-size state. The honest question:
does SV-Attention's adaptive, bounded support set retain more distinct items
than DeltaNet's fixed d x d_v state, while staying far below full-context cost?
"""
from __future__ import annotations

import numpy as np
import torch

import svattn.selective_copy as sc
from svattn.selective_copy import (SVAttnModel, SoftmaxModel, DeltaNetModel,
                                   LinearAttnModel, train, evaluate, sv_state_size)

STATE_D = 16   # fixed-state dim for DeltaNet / linear (their capacity knob)


def make_ortho(n_data, n_filler, d_in, dv_in, rng, blank_noise=0.02):
    """Near-orthogonal content keys (d_in must be >= n_data)."""
    base = np.eye(d_in)[:n_data]
    contents = base + 0.02 * rng.randn(n_data, d_in)
    values = rng.randn(n_data, dv_in)
    filler_c = 0.02 * rng.randn(n_filler, d_in)
    filler_v = np.zeros((n_filler, dv_in))
    X = np.vstack([contents, filler_c]); Vf = np.vstack([values, filler_v])
    j = rng.randint(n_data)
    q = contents[j:j + 1] + 0.02 * rng.randn(1, d_in)
    perm = rng.permutation(len(X))
    return (torch.tensor(X[perm]), torch.tensor(Vf[perm]), torch.tensor(q),
            torch.tensor(values), torch.tensor(values[j:j + 1]), j)


def main():
    sc.make_instance = make_ortho        # fair, near-orthogonal task
    torch.manual_seed(0)
    F = 20
    dv = 8
    print(f"Fair orthogonal-key recall, {F} filler tokens, fixed state dim={STATE_D} "
          f"for DeltaNet/linear.\n")
    print(f"{'#items':>7}{'d_in':>6}{'SV-Attn':>10}{'softmax':>10}{'DeltaNet':>10}{'linear':>10}{'SV state':>11}")
    for n_data in (8, 16, 24, 40):
        d_in = max(48, n_data + 8)
        cfg = dict(steps=400, seqs=4, lr=0.02)
        sv = train(SVAttnModel(d_in, dv, d=STATE_D, dv=dv, C=0.6, kpar=1.5), n_data, F, d_in, dv, **cfg)
        sm = train(SoftmaxModel(d_in, dv, d=STATE_D, dv=dv), n_data, F, d_in, dv, **cfg)
        dn = train(DeltaNetModel(d_in, dv, d=STATE_D, dv=dv), n_data, F, d_in, dv, **cfg)
        lin = train(LinearAttnModel(d_in, dv, d=STATE_D, dv=dv), n_data, F, d_in, dv, **cfg)
        a = {n: evaluate(m, n_data, F, d_in, dv) for n, m in
             [("sv", sv), ("sm", sm), ("dn", dn), ("lin", lin)]}
        st = sv_state_size(SVAttnModel(d_in, dv, d=STATE_D, dv=dv, C=0.6, kpar=1.5), n_data, F, d_in, dv)
        print(f"{n_data:>7}{d_in:>6}{a['sv']:>10.3f}{a['sm']:>10.3f}{a['dn']:>10.3f}{a['lin']:>10.3f}"
              f"{st:>7.1f}/{n_data + F}", flush=True)
    print("\nFair read: with separable keys all methods can retrieve; the question is "
          "whether the FIXED-state baselines fade as distinct items exceed their capacity, "
          "while SV-Attention's bounded support adapts.")


if __name__ == "__main__":
    main()
