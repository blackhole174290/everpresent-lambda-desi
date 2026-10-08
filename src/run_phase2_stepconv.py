"""フェーズ2：完走率と生存者統計の刻み幅依存（ln a 等間隔、z=1e4 開始、厳密な体積）。
使い方: python3 src/run_phase2_stepconv.py <alpha> <N> n1 n2 ...（結果は JSON に追記）"""
import numpy as np, sys, json, os, time
sys.path.insert(0, 'src')
import everpresent as E
al, N = float(sys.argv[1]), int(sys.argv[2]); ns = [int(x) for x in sys.argv[3:]]
fn = 'results/phase2_stepconv.json'
db = json.load(open(fn)) if os.path.exists(fn) else {}
for n in ns:
    t = time.time(); fs = []; hs = []; oms = []
    for b in range(0, N, 2000):
        r = E.realize(0.14, al, min(2000, N - b), seed=int(1e6 * al) + n + b, grid='loga', n=n,
                      a_init=1e-4, keep_every=10**7, n_sub=2)
        fs.append(r['alive']); hs.append(r['h'][r['alive']]); oms.append(r['OmL0'][r['alive']])
    al_ = np.concatenate(fs); h = 100 * np.concatenate(hs); om = np.concatenate(oms)
    f = al_.mean(); se = np.sqrt(f * (1 - f) / len(al_))
    db[f"{al}_{n}"] = dict(alpha=al, n=n, N=len(al_), complete=f, se=se, H0_mean=float(h.mean()),
                          OmL_std=float(om.std()), frac_H0_67_72_all=float(np.mean((h > 67) & (h < 72)) * f))
    print(f"α={al} n={n:6d}: 完走率 {f:.4f} ± {se:.4f}, 生存者の H0 平均 {h.mean():.1f}, Ω_Λ0 の幅 {om.std():.3f}, "
          f"全体に対する H0∈[67,72] {db[f'{al}_{n}']['frac_H0_67_72_all']:.4f}  [{time.time()-t:.0f}s]")
    json.dump(db, open(fn, 'w'), indent=1)
