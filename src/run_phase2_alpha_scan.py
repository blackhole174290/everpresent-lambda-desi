"""フェーズ2：α に対する破綻率・H0 分布・Ω_Λ0 分布（ω_m h² = 0.14）。Das et al. II §4.1 の記述と比べる。"""
import numpy as np, sys, time, json
sys.path.insert(0, 'src')
import everpresent as E
alphas = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.020, 0.025]
N = 4000; out = []
t = time.time()
print(" α      破綻率   生存のうち: h中央値  h∈[0.6,0.8]の割合  Ω_Λ0中央値  Ω_Λ0>0の割合   全体に対する h∈[0.6,0.8]")
for i, al in enumerate(alphas):
    r = E.realize(0.14, al, N, seed=1000 + i, keep_every=400)
    alive = r['alive']; h = r['h'][alive]; oml = r['OmL0'][alive]
    inb = np.mean((h > 0.6) & (h < 0.8)) if len(h) else 0
    row = dict(alpha=al, crash=1 - alive.mean(), h_med=float(np.median(h)) if len(h) else None,
               frac_h_obs_alive=float(inb), OmL_med=float(np.median(oml)) if len(h) else None,
               frac_OmL_pos=float(np.mean(oml > 0)) if len(h) else None, frac_h_obs_all=float(inb * alive.mean()))
    out.append(row)
    print(f"{al:.3f}   {row['crash']:.3f}    {row['h_med']:.3f}        {inb:.3f}            {row['OmL_med']:+.3f}      {row['frac_OmL_pos']:.3f}          {row['frac_h_obs_all']:.3f}")
json.dump(out, open('results/phase2_alpha_scan.json', 'w'), indent=1)
print(f"計算時間 {time.time()-t:.0f}s（{len(alphas)}×{N} 実現）")
