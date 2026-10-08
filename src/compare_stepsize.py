"""フェーズ4：刻み幅の頑健性。n=2000 と n=8000 で、完走率と Δχ² の分布（全実現比）を比べる。"""
import numpy as np, glob, json
ref = {k: v[0] for k, v in json.load(open('results/phase3_late_summary.json'))['ref'].items()}
combos = {'BAO': ['BAO'], 'BAO+U3': ['BAO', 'U3'], 'BAO+PP': ['BAO', 'PP'], 'BAO+DY5': ['BAO', 'DY5']}
out = {}
for al in (0.008, 0.009):
    for n, d in ((2000, 'results/p3late'), (8000, 'results/p3late_n8000')):
        D = [np.load(f) for f in glob.glob(f'{d}/a{al:.3f}_*.npz')]
        N = sum(len(x['alive']) for x in D); comp = sum(x['alive'].sum() for x in D) / N
        row = {'N': N, 'complete': comp}
        txt = []
        for name, parts in combos.items():
            c = np.concatenate([sum(x[p] for p in parts) for x in D]) - ref[name]; c = c[~np.isnan(c)]
            row[name] = {'best': float(c.min()), **{f'f_lt{t}': float((c < t).sum() / N) for t in (9, 25, 60)}}
            txt.append(f"{name} 最良 {c.min():+.1f}, <25: {(c<25).sum()/N:.1e}, <60: {(c<60).sum()/N:.1e}")
        out[f'{al}_{n}'] = row
        print(f"α={al} n={n} (N={N}): 完走率 {comp:.3f}\n   " + "\n   ".join(txt))
json.dump(out, open('results/phase4_stepsize.json', 'w'), indent=1)
