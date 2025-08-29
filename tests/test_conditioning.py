import torch, numpy as np
from tests.test_utils import build_grid_and_spectra, save_npz, ensure_dir

def main():
    E, w, S, I0, B = build_grid_and_spectra()
    W = (w * 0.5*(S[0]+S[1])).clamp_min(1e-12)
    G = (B * W[:,None]).T @ B
    Gs = G + 1e-8*torch.eye(G.shape[0], device=G.device)
    ev = torch.linalg.eigvalsh(Gs)
    cond = float((ev.max()/ev.min()).item())
    offdiag = (G - torch.diag(torch.diag(G))).abs()
    off_mean = float(offdiag.mean().item())
    diag_mean = float(torch.diag(G).mean().item())
    ratio = off_mean / max(diag_mean, 1e-12)
    print(f"[A1] Gram:\n{G.detach().cpu().numpy()}")
    print(f"[A1] cond(G)={cond:.2e}, mean|offdiag|/mean(diag)={ratio:.3f}")
    save_npz("artifacts/conditioning/gram_before.npz", G=G)
    ok = (cond < 1e2) and (ratio < 0.10)
    print(f"\nA1 SUMMARY: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 3

if __name__ == "__main__":
    raise SystemExit(main())