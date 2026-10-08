"""フェーズ2：Ω_Λ の自己相関（Das et al. I 図8）。α=0.01、ω_m h²=0.14、生存した実現のみ。
各実現で λ₁ について平均した ⟨Ω_Λ(λ₁)Ω_Λ(λ₁+Δλ)⟩ を、実現について平均し、Δλ=0 で1に規格化。
論文の当てはめ：exp(−3.2 Δλ^0.48)。"""
import numpy as np, sys, json
from scipy.optimize import curve_fit
sys.path.insert(0, 'src')
import everpresent as E
n = 2000
r = E.realize(0.14, 0.01, 6000, seed=77, grid='loga', n=n, a_init=1e-4, keep_every=1, n_sub=2)
al = r['alive']; om = (r['Lam'] / 3 / r['H']**2)[:, al]          # (時刻, 実現)
lna = np.log(r['a']); dl = lna[1] - lna[0]
print(f"生存した実現 {al.sum()} 本、Δ ln a = {dl:.5f}")
sel = lna >= np.log(1e-3)                                          # 物質優勢期中心（放射期の端の影響を避ける）
X = om[sel]; T = len(X)
lags = np.arange(0, int(3.0 / dl))
C = np.array([np.mean(np.mean(X[:T - k] * X[k:], axis=0)) for k in lags])
C /= C[0]; D = lags * dl
f2 = lambda x, c1, c2: np.exp(-c1 * x**c2)
m = D > 0
(p1, p2), _ = curve_fit(f2, D[m], C[m], p0=[3, 0.5])
c_exp, _ = curve_fit(lambda x, c: np.exp(-c * x), D[m], C[m], p0=[3])
print(f"当てはめ exp(−c₁Δλ^c₂): c₁ = {p1:.2f}, c₂ = {p2:.2f}   （Das et al.: c₁ = 3.2, c₂ = 0.48）")
print(f"当てはめ exp(−cΔλ): c = {c_exp[0]:.2f}   （Das et al. の解析的な見積もり: c = 3）")
for d in [0.1, 0.3, 0.7, 1.0, 2.0]:
    i = np.argmin(abs(D - d)); print(f"  Δλ={D[i]:.2f}: 相関 {C[i]:.3f}  （論文式 {f2(D[i], 3.2, 0.48):.3f}）")
json.dump({'D': D.tolist(), 'C': C.tolist(), 'c1': p1, 'c2': p2, 'c_exp': float(c_exp[0]), 'n_alive': int(al.sum())},
          open('results/phase2_autocorr.json', 'w'))
