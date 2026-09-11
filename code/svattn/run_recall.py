"""Compare SV-Attention vs matched softmax attention on associative recall.

Run (needs torch; uses the ferg venv):
    PYTHONPATH=. \
        ./ferg/.venv/bin/python -m svattn.run_recall
"""
import torch

from svattn.recall import (SVAttnRecall, SoftmaxAttnRecall, RBFAttnRecall,
                           train_recall, eval_recall)


def main():
    torch.manual_seed(0)
    cfg = dict(n_pairs=10, d_in=5, dv_in=3, steps=500, seqs_per_step=4)

    print("Associative recall (10 pairs, d_in=5, dv=3):\n")
    sv = SVAttnRecall(d_in=cfg["d_in"], dv_in=cfg["dv_in"], C=0.3, kpar=1.5)
    sv_losses = train_recall(sv, **cfg, lr=0.03, seed=0)
    sv_acc, sv_mse = eval_recall(sv, n_pairs=cfg["n_pairs"], d_in=cfg["d_in"], dv_in=cfg["dv_in"])
    mask_frac = float(sv.attn.support_mask(
        torch.randn(cfg["n_pairs"], 5, dtype=torch.float64) @ sv.Wk).float().mean())

    rbf = RBFAttnRecall(d_in=cfg["d_in"], dv_in=cfg["dv_in"], kpar=1.5)
    rbf_losses = train_recall(rbf, **cfg, lr=0.03, seed=0)
    rbf_acc, rbf_mse = eval_recall(rbf, n_pairs=cfg["n_pairs"], d_in=cfg["d_in"], dv_in=cfg["dv_in"])

    sm = SoftmaxAttnRecall(d_in=cfg["d_in"], dv_in=cfg["dv_in"])
    sm_losses = train_recall(sm, **cfg, lr=0.03, seed=0)
    sm_acc, sm_mse = eval_recall(sm, n_pairs=cfg["n_pairs"], d_in=cfg["d_in"], dv_in=cfg["dv_in"])

    print(f"{'model':<28}{'train loss':>12}{'recall@1':>11}{'eval MSE':>11}")
    print(f"{'SV-Attention (ours)':<28}{sv_losses[-1]:>12.4f}{sv_acc:>11.3f}{sv_mse:>11.4f}")
    print(f"{'RBF attention (uniform a)':<28}{rbf_losses[-1]:>12.4f}{rbf_acc:>11.3f}{rbf_mse:>11.4f}")
    print(f"{'softmax (dot-product)':<28}{sm_losses[-1]:>12.4f}{sm_acc:>11.3f}{sm_mse:>11.4f}")
    print(f"\nSV-Attention support fraction (nonzero-weight keys): {mask_frac:.2f} "
          f"-> {(1-mask_frac)*100:.0f}% of context tokens are certified-inert (zero weight)")


if __name__ == "__main__":
    main()
