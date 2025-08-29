import os
import sys
import numpy as np
import torch
from tests.test_utils import build_grid_and_spectra, beer_lambert, save_npz, ensure_dir


def main():
    E, w, S, I0, B = build_grid_and_spectra()
    device = E.device
    logs = []
    arts = ensure_dir("tests/artifacts/basis")

    # ---------- BF1: raw basis sanity ----------
    names = ["PE","Compton"] + (["Kedge"] if B.shape[1]==3 else [])
    ok_all = True
    for j, nm in enumerate(names):
        bj = B[:, j]
        finite = torch.isfinite(bj).all().item()
        pos    = (bj > 0).all().item()
        mn, mx = float(bj.min()), float(bj.max())
        dr     = mx / max(mn, 1e-30)
        sgn    = float(torch.median(torch.sign(torch.diff(bj))).item())
        logs.append(f"[BF1] {nm}: finite={finite} >0={pos} min={mn:.3e} max={mx:.3e} dynRange={dr:.3e} med sign(diff)={sgn:+.0f}")
        # PE should be strongly decreasing and high DR
        if nm=="PE" and not (finite and pos and sgn<=0 and dr>50):
            ok_all=False
        if nm=="Compton" and not (finite and pos):
            ok_all=False
    if B.shape[1]==3:
        kidx = torch.argmax(B[:,2])
        logs.append(f"[BF1] Kedge max @ {float(E[kidx]):.1f} keV")
    print("\n".join(logs))

    # ---------- BF2: weighted coverage + gap ----------
    S_avg = 0.5*(S[0]+S[1])
    W = (S_avg * w).clamp_min(1e-12)
    M = B * W[:, None]
    Mnorm = M / (M.max(dim=0).values.clamp_min(1e-12)[None,:])
    def pct_energy_window(col, plo=5, phi=95):
        v = col.detach().cpu().numpy()
        lo, hi = np.percentile(v, [plo, phi])
        mask = (v>=lo) & (v<=hi)
        e = E.detach().cpu().numpy()[mask]
        return (float(e.min()) if e.size else None, float(e.max()) if e.size else None)
    ok_cov = True
    for j, nm in enumerate(names):
        elo, ehi = pct_energy_window(M[:,j])
        tot = float(M[:,j].sum().cpu())
        print(f"[BF2] {nm}: 5–95% coverage ≈ [{elo:.1f}, {ehi:.1f}] keV, weighted mass={tot:.3e}")
    thr=0.10
    weak_all = (Mnorm < thr).all(dim=1)
    gap_bins = int(weak_all.sum().cpu())
    if gap_bins>0:
        eg = E[weak_all].cpu().numpy()
        wwidth = float((weak_all*w).sum().cpu())
        print(f"[BF2] GAP: {gap_bins} bins, ≈{wwidth:.2f} keV around [{eg.min():.1f},{eg.max():.1f}] keV")
        ok_cov=False
    else:
        print("[BF2] GAP: none")

    # ---------- BF3: cross-over ----------
    ok_x = True
    if B.shape[1]>=2:
        r = (B[:,0] / B[:,1]).clamp_min(1e-30)
        idx = torch.argmin((r-1).abs())
        E_cross = float(E[idx])
        print(f"[BF3] PE≈Comp around {E_cross:.1f} keV, ratio={float(r[idx]):.3f}")
        if not (float(E.min())+1.0 <= E_cross <= float(E.max())-1.0):
            ok_x=False
        topk = torch.topk(0.5*(S[0]+S[1]), k=min(3, S.shape[1])).indices
        vals=[(float(E[i]), float(r[i])) for i in topk]
        print(f"[BF3] ratio at S_avg top bins: {vals}")

    # ---------- BF4: sliding-window condition ----------
    ok_cond=True
    if B.shape[1]>=2:
        win = 7
        conds=[]
        for s in range(0, E.numel()-win+1):
            sl=slice(s,s+win)
            Be=B[sl,:]; Wsl=(W[sl]).clamp_min(1e-12)
            G=(Be*Wsl[:,None]).T @ Be
            G=G+1e-9*torch.eye(G.shape[0], device=device)
            ev=torch.linalg.eigvalsh(G)
            conds.append(float((ev.max()/ev.min()).item()))
        bad = [c for c in conds if c>1e3]
        print(f"[BF4] sliding cond>1e3 windows: {len(bad)}/{len(conds)}")
        if len(bad) > 0.1*len(conds): ok_cond=False

    save_npz(f"{arts}/basis_curves.npz", E=E, B=B, S=S, w=w)

    passed = ok_all and ok_cov and ok_x and ok_cond
    print(f"\nBF SUMMARY: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2

if __name__ == "__main__":
    raise SystemExit(main())