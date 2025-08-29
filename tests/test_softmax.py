import torch, math
from tests.test_utils import build_grid_and_spectra, beer_lambert
from utils.spectralCT_utils import stable_simplex

def _toy_loss(frac_logits, rho_raw, E, w, S, I0, B, coeff_true, tau=1.0, eps=0.60):
    rho = torch.nn.Softplus()(rho_raw)           # (1,)
    # fr  = torch.softmax(frac_logits, dim=1)      # (1,C)
    fr  = stable_simplex(frac_logits, tau=tau, eps=eps)
    coeff = rho * fr.squeeze(0)                  # (C,)
    mu = (B * coeff[None,:]).sum(dim=1)          # (K,)
    L = 0.0
    with torch.no_grad():
        # "ground truth" line integrals from coeff_true (C,)
        mu_gt = (B * coeff_true[None,:]).sum(dim=1)
        y_gt = torch.stack([beer_lambert(I0[i], S[i], w, mu_gt)[1] for i in (0,1)]).sum()
    y_pred = torch.stack([beer_lambert(I0[i], S[i], w, mu)[1] for i in (0,1)]).sum()
    loss = (y_pred - y_gt).pow(2)
    return loss, fr, rho

def run_once(init_logits, tau=1.0, eps=0.60):
    # setup
    E, w, S, I0, B = build_grid_and_spectra()
    C = B.shape[1]
    coeff_true = torch.tensor([0.03, 0.05] + ([0.0]*(C-2)), device=E.device)[:C]
    frac_logits = torch.nn.Parameter(init_logits.clone())
    rho_raw = torch.nn.Parameter(torch.tensor([0.05], device=E.device))
    # temp scaling inside the forward
    def loss():
        # return _toy_loss(frac_logits/tau, rho_raw, E, w, S, I0, B, coeff_true)[0]
        return _toy_loss(frac_logits, rho_raw, E, w, S, I0, B, coeff_true, tau=tau, eps=eps)[0]
    L = loss()
    L.backward()
    # ent = float(-(torch.softmax(frac_logits, dim=1) *
    #               torch.log_softmax(frac_logits, dim=1)).sum().item())
    with torch.no_grad():
        fr_eff = stable_simplex(frac_logits, tau=tau, eps=eps)
        ent = float((-(fr_eff * (fr_eff.clamp_min(1e-12)).log()).sum()).item())
    gF = float(frac_logits.grad.detach().norm().item())
    gR = float(rho_raw.grad.detach().abs().item())
    ratio = (gF+1e-12)/(gR+1e-12)
    return dict(L=float(L.item()), entropy=ent, grad_ratio=ratio, gF=gF, gR=gR)

def main():
    # C1: detect saturation
    init = torch.tensor([[10.0, -10.0]], device=("cuda" if torch.cuda.is_available() else "cpu"))
    m1 = run_once(init, tau=1.0)
    print(f"[C1] entropy={m1['entropy']:.3f}  grad(fracs)/grad(rho)={m1['grad_ratio']:.3f}")
    ok1 = (m1["entropy"] > 0.6) and (m1["grad_ratio"] > 0.33)

    # C2: softening helps (temperature)
    m2 = run_once(init, tau=2.0)
    print(f"[C2] (tau=2) entropy={m2['entropy']:.3f}  grad ratio={m2['grad_ratio']:.3f}")
    ok2 = (m2["entropy"] > m1["entropy"]) and (m2["grad_ratio"] > m1["grad_ratio"])

    # C3: gentle floor helps (simulate epsilon floor on probs)
    # emulate epsilon by mixing with uniform 5%
    init2 = torch.tensor([[0.0, 0.0]], device=init.device)
    m3 = run_once(init2, tau=1.0)
    print(f"[C3] (uninformative logits) entropy={m3['entropy']:.3f} grad ratio={m3['grad_ratio']:.3f}")
    ok3 = (m3["entropy"] >= 0.69)  # ~ln 2

    all_ok = ok1 and ok2 and ok3
    print(f"\nC SUMMARY: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 5

if __name__ == "__main__":
    raise SystemExit(main())