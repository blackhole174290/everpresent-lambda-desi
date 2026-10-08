"""フェーズ2：Das et al. I §5.1（図2）の再現。ω_m h²=0.14、α=0.005, 0.01, 0.012。"""
import numpy as np, sys, time, json
sys.path.insert(0, 'src')
import everpresent as E
target = {0.005: 0.9990, 0.010: 0.3228, 0.012: 0.0535}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
VOL = sys.argv[2] if len(sys.argv) > 2 else 'das'
print('体積の扱い:', VOL)
res = {}
for i, al in enumerate(target):
    t = time.time()
    r = E.realize(0.14, al, N, seed=2000 + i, grid='das', keep_every=1000, volume=VOL)
    al_ = r['alive']; h = 100 * r['h'][al_]; om = r['OmL0'][al_]
    f = al_.mean(); se = np.sqrt(f * (1 - f) / N)
    res[al] = dict(complete=float(f), complete_se=float(se), target=target[al],
                   H0_mean=float(h.mean()), H0_std=float(h.std()),
                   frac_H0_67_72=float(np.mean((h > 67) & (h < 72))),
                   OmL_mean=float(om.mean()), OmL_std=float(om.std()),
                   H04V_median=float(np.median((r['h']**4 * r['V0'])[al_])))
    print(f"α={al}: 完走率 {f:.4f} ± {se:.4f}（Das: {target[al]:.4f}）, H0 平均 {h.mean():.1f} ± {h.std():.1f} km/s/Mpc, "
          f"H0∈[67,72] の割合 {res[al]['frac_H0_67_72']:.4f}, Ω_Λ0 平均 {om.mean():+.3f} 幅 {om.std():.3f}, "
          f"H0⁴V 中央値 {res[al]['H04V_median']:.4f}  [{time.time()-t:.0f}s]")
json.dump(res, open(f'results/phase2_das_repro_{VOL}.json', 'w'), indent=1)
