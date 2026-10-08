"""フェーズ3（CMB非依存）の集計：各 α の生存率（全実現に対する割合）と最良 Δχ²。"""
import numpy as np, sys, glob, json
from scipy.optimize import minimize_scalar
sys.path.insert(0, 'src')
from late_chi2 import LateShape
from cosmology import OMEGA_R_H2
LS = LateShape()
a = np.exp(np.linspace(np.log(0.27), 0, 400))
def lcdm(Om, h=0.68):
    return LS.chi2(a, np.sqrt(Om / a**3 + OMEGA_R_H2 / h**2 / a**4 + 1 - Om)[:, None])
combos = {'BAO': ['BAO'], 'BAO+U3': ['BAO', 'U3'], 'BAO+PP': ['BAO', 'PP'], 'BAO+DY5': ['BAO', 'DY5']}
ref = {}
for name, parts in combos.items():
    r = minimize_scalar(lambda Om: sum(lcdm(Om)[p][0] for p in parts), bounds=(0.2, 0.45), method='bounded')
    ref[name] = (float(r.fun), float(r.x))
print("ΛCDM 基準（最良 χ², Ω_m）:", {k: (round(v[0], 2), round(v[1], 3)) for k, v in ref.items()})
out = {'ref': ref, 'alpha': {}}
files = sorted(glob.glob('results/p3late/*.npz'))
alphas = sorted({float(f.split('/a')[1][:5]) for f in files})
print("\n α     全実現   完走率 | 最良Δχ²: BAO  +U3  +PP  +DY5 | Δχ²<9 の割合（全実現比）: BAO  +U3  +PP  +DY5")
for al in alphas:
    D = [np.load(f) for f in files if f'/a{al:.3f}_' in f]
    N = sum(len(d['alive']) for d in D); comp = sum(d['alive'].sum() for d in D) / N
    row = {'N': N, 'complete': comp}
    best, f9 = [], []
    for name, parts in combos.items():
        c = np.concatenate([sum(d[p] for p in parts) for d in D]) - ref[name][0]
        c = c[~np.isnan(c)]
        row[name] = {'best': float(c.min()), **{f'n_lt{t}': int((c < t).sum()) for t in (0, 4, 9, 25)}}
        best.append(c.min()); f9.append((c < 9).sum() / N)
    out['alpha'][al] = row
    print(f"{al:.3f}  {N:6d}  {comp:.3f} | {best[0]:+6.1f} {best[1]:+6.1f} {best[2]:+6.1f} {best[3]:+6.1f} | "
          + "  ".join(f"{x:.1e}" for x in f9))
tot = sum(r['N'] for r in out['alpha'].values())
print(f"\n合計 {tot} 実現。Δχ²<t となった実現の総数:")
for name in combos:
    print(f"  {name:8s}: " + ", ".join(f"<{t}: {sum(r[name][f'n_lt{t}'] for r in out['alpha'].values())}" for t in (0, 4, 9, 25)))
json.dump(out, open('results/phase3_late_summary.json', 'w'), indent=1)
