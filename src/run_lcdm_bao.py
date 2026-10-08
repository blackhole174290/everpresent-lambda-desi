"""フェーズ1-a：DESI DR2 BAO 単独で ΛCDM を当てはめ、公表値と比較。
平坦事前分布の2次元格子で事後分布を計算（MCMC 不要の規模）。"""
import numpy as np, sys
sys.path.insert(0, 'src')
from likelihood import DesiBAO
bao = DesiBAO()
Om = np.linspace(0.25, 0.35, 201)
hrd = np.linspace(98, 105, 201)
chi = np.array([[bao.chi2(o, r) for r in hrd] for o in Om])
P = np.exp(-0.5 * (chi - chi.min())); P /= P.sum()
def stats(x, p): m = (x * p).sum(); return m, np.sqrt(((x - m)**2 * p).sum())
mO, sO = stats(Om, P.sum(1)); mR, sR = stats(hrd, P.sum(0))
i, j = np.unravel_index(chi.argmin(), chi.shape)
print(f"最良値: Om={Om[i]:.4f}, hrd={hrd[j]:.2f}, chi2_min={chi.min():.2f} (データ点13, 自由度11)")
print(f"事後平均: Om = {mO:.4f} ± {sO:.4f}   (DESI公表 0.2975 ± 0.0086)")
print(f"事後平均: hrd = {mR:.2f} ± {sR:.2f} Mpc (DESI公表 101.54 ± 0.73)")
print(f"ずれ: Om {(mO-0.2975)/0.0086:+.2f}σ, hrd {(mR-101.54)/0.73:+.2f}σ")
np.savez('results/lcdm_bao_grid.npz', Om=Om, hrd=hrd, chi2=chi)
