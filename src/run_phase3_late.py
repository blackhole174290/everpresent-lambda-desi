"""フェーズ3：CMB に依存しない判定。DESI DR2 BAO（相対）＋超新星で、各実現の χ² を ΛCDM の最良値と比べる。
設定：ω_m=0.142（z<2.5 の形は種と α でほぼ決まる）、ln a 等間隔 n=2000、z=10⁴ 開始、厳密な体積、S₀=0。
5000 実現ずつ results/p3late/ に保存し、300 秒制限の中で繰り返し実行して積み上げる。
使い方: python3 src/run_phase3_late.py <目標実現数/α>"""
import numpy as np, sys, os, time, glob
TIME_LIMIT = float(os.environ.get('TIME_LIMIT', 1e9))  # 秒。実行時間に上限がある環境では設定する（例: 200）
sys.path.insert(0, 'src')
import everpresent as E
from late_chi2 import LateShape
T0 = time.time(); NT = int(sys.argv[1])
NSTEP = int(os.environ.get('NSTEP', 2000))          # 刻み数（頑健性テスト用に環境変数で変更）
OUTD = 'results/p3late' if NSTEP == 2000 else f'results/p3late_n{NSTEP}'
os.makedirs(OUTD, exist_ok=True)
alphas = [float(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [0.004, 0.006, 0.008, 0.010, 0.012, 0.014]
LS = None; B = 5000
# 注：α=0.004〜0.014 の最初の 12 万実現は旧式の種（700000+1000·ia+b）で生成済み。ファイル名が同じものは再計算しない
for ia, al in enumerate(alphas):
    for b in range(NT // B):
        fn = f'{OUTD}/a{al:.3f}_b{b:02d}.npz'
        if os.path.exists(fn): continue
        if time.time() - T0 > TIME_LIMIT: print('時間切れ、続きは再実行で'); sys.exit()
        if LS is None: LS = LateShape()
        seed = 700000 + int(round(al * 1e6)) * 100 + b + (0 if NSTEP == 2000 else 50000 + NSTEP)   # α・batch・刻みから一意
        r = E.realize(0.142, al, B, seed=seed, grid='loga', n=NSTEP, a_init=1e-4, keep_every=1, n_sub=2, rec_amin=0.27)
        alive = r['alive']; idx = np.where(alive)[0]
        res = {'alive': alive, 'seed': seed, 'h': r['h'], 'OmL0': r['OmL0']}
        if len(idx):
            c = LS.chi2(r['a'], r['H'][:, idx])
            for k, v in c.items():
                full = np.full(B, np.nan); full[idx] = v; res[k] = full
        # 最良の数実現は膨張史も保存（後で図にする）
        if len(idx):
            best = idx[np.argsort(res['BAO'][idx])[:5]]
            res['best_idx'] = best; res['best_H'] = r['H'][:, best]; res['a_rec'] = r['a']
        np.savez_compressed(fn, **res)
        print(f"α={al} batch {b}: 生存 {alive.mean():.3f}, BAO 最小 χ² {np.nanmin(res.get('BAO', [np.nan])):.1f}  [{time.time()-T0:.0f}s]")
print('全バッチ完了')
