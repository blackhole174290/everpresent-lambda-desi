"""フェーズ1-b：超新星単独で平坦 ΛCDM の Ω_m を再現。"""
import numpy as np, sys
sys.path.insert(0, 'src')
from likelihood import PantheonPlus, DESY5, Union3
targets = {'Pantheon+': (0.334, 0.018), 'DES-SN5YR': (0.352, 0.017), 'Union3': (0.356, 0.027)}
Om = np.linspace(0.15, 0.55, 401)
out = {}
for cls in (PantheonPlus, DESY5, Union3):
    sn = cls()
    chi = np.array([sn.chi2(o) for o in Om])
    P = np.exp(-0.5 * (chi - chi.min())); P /= P.sum()
    m = (Om * P).sum(); s = np.sqrt(((Om - m)**2 * P).sum())
    t, ts = targets[sn.name]
    print(f"    Om = {m:.4f} ± {s:.4f}  (公表 {t} ± {ts}, ずれ {(m-t)/ts:+.2f}σ), 最良値 {Om[chi.argmin()]:.4f}")
    out[sn.name] = chi
np.savez('results/lcdm_sn_grid.npz', Om=Om, **{k.replace('+','plus').replace('-','_'): v for k, v in out.items()})
