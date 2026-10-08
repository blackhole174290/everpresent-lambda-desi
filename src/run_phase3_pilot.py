"""フェーズ3の前哨戦（粗いパイロット）：ω_b, ω_bc を CMB 平均値に固定し、各実現の χ² を ΛCDM の最良値と比べる。
実現ごとのパラメータ最適化はしていない（フェーズ3本番で行う）。"""
import numpy as np, sys, json, time
from scipy.optimize import minimize
sys.path.insert(0, 'src')
import everpresent as E
from everpresent_bg import EverpresentBackground
from background import Background
import likelihood as L
bao, cmb, U3 = L.DesiBAOAbs(), L.CMBCompressed(), L.Union3()
wb, wbc = cmb.mean[1], cmb.mean[2]
wm_model = wbc + 0.06 / 93.14
# ΛCDM の基準 χ²（同じデータ、ω_b・ω_bc・h を最適化）
def chiL(p, sn):
    bg = Background(*p); c = bao.chi2_bg(bg) + cmb.chi2_bg(bg)
    return c + (L.sn_chi2_bg(U3, bg) if sn else 0)
ref = {sn: minimize(chiL, [0.0223, 0.1400, 0.684], args=(sn,), method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-7, 'maxiter': 5000}).fun for sn in (False, True)}
print(f"ΛCDM 基準 χ²: DESI+CMB = {ref[False]:.2f}, DESI+CMB+Union3 = {ref[True]:.2f}")
out = {'ref': ref, 'rows': []}
for i, al in enumerate([0.006, 0.008, 0.010, 0.012, 0.015]):
    t = time.time()
    r = E.realize(wm_model, al, 3000, seed=9000 + i, grid='loga', n=2000, a_init=1e-4, keep_every=4, n_sub=2)
    idx = np.where(r['alive'])[0]
    c1, c2, hs, oml, frec = [], [], [], [], []
    for j in idx:
        bg = EverpresentBackground(wb, wbc, r['a'], r['Lam'][:, j] / 3)
        c = bao.chi2_bg(bg) + cmb.chi2_bg(bg)
        c1.append(c); c2.append(c + L.sn_chi2_bg(U3, bg)); hs.append(bg.h); oml.append(r['OmL0'][j]); frec.append(bg.omL_rec)
    c1, c2, hs, frec = map(np.array, (c1, c2, hs, frec))
    d1, d2 = c1 - ref[False], c2 - ref[True]
    N = len(r['alive'])
    row = dict(alpha=al, complete=len(idx) / N,
               frac_all_d1={th: float(np.sum(d1 < th) / N) for th in (0, 4, 9, 25)},
               frac_all_d2={th: float(np.sum(d2 < th) / N) for th in (0, 4, 9, 25)},
               best_d1=float(d1.min()), best_d2=float(d2.min()), h_med=float(np.median(hs)),
               frec_med=float(np.median(np.abs(frec))))
    out['rows'].append(row)
    print(f"α={al}: 完走 {row['complete']:.3f}, 最良 Δχ² (DESI+CMB) {d1.min():+.1f}, (+U3) {d2.min():+.1f}; "
          f"全実現に対する割合 Δχ²<9: {row['frac_all_d1'][9]:.4f} / {row['frac_all_d2'][9]:.4f}, "
          f"Δχ²<25: {row['frac_all_d1'][25]:.4f} / {row['frac_all_d2'][25]:.4f}; "
          f"h 中央値 {row['h_med']:.3f}, 再結合期の |Λ割合| 中央値 {row['frec_med']:.3f}  [{time.time()-t:.0f}s]")
json.dump(out, open('results/phase3_pilot.json', 'w'), indent=1)
