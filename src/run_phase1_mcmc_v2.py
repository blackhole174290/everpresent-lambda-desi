"""フェーズ1-d：SN を含む w0wa ケース。ウォーカーを束ねて評価し、途中保存・再開に対応。
使い方: python3 src/run_phase1_mcmc_v2.py <ケース> <総ステップ数>（300秒制限のため、完了まで繰り返し実行）"""
import numpy as np, sys, emcee, json, time, os
TIME_LIMIT = float(os.environ.get('TIME_LIMIT', 1e9))  # 秒。実行時間に上限がある環境では設定する（例: 230）
sys.path.insert(0, 'src')
from background import Background, Z_LOW
import likelihood as L
T0 = time.time(); case, NTOT = sys.argv[1], int(sys.argv[2])
kind, S_name = case.rsplit('_', 1)            # 例: w0wa_sn_PP → ('w0wa_sn', 'PP')
S = {'PP': L.PantheonPlus, 'U3': L.Union3, 'DY5': L.DESY5}[S_name]()
bao_rel, bao_abs, cmb = L.DesiBAO(), L.DesiBAOAbs(), L.CMBCompressed()
ok = lambda w0, wa: -3 <= w0 <= 1 and -3 <= wa <= 2 and w0 + wa < 0

if kind == 'w0wa_sn':
    names, p0 = ['Om', 'hrd', 'w0', 'wa'], [0.31, 100.0, -0.8, -0.5]
    def lnp_batch(P):
        out = np.full(len(P), -np.inf); idx, Es, base = [], [], []
        for i, (Om, hrd, w0, wa) in enumerate(P):
            if not (0.01 < Om < 0.99 and 10 < hrd < 1000 and ok(w0, wa)): continue
            idx.append(i); base.append(bao_rel.chi2(Om, hrd, w0, wa))
            Es.append(lambda z, Om=Om, w0=w0, wa=wa: np.sqrt(Om*(1+z)**3 + (1-Om)*(1+z)**(3*(1+w0+wa))*np.exp(-3*wa*z/(1+z))))
        if idx: out[idx] = -0.5 * (np.array(base) + L.sn_chi2_batch(S, Es))
        return out
elif kind == 'w0wa_cmb':
    names, p0 = ['wb', 'wbc', 'h', 'w0', 'wa'], [0.0222, 0.1414, 0.67, -0.75, -0.8]
    def lnp_batch(P):
        out = np.full(len(P), -np.inf); idx, Es, base = [], [], []
        for i, (wb, wbc, h, w0, wa) in enumerate(P):
            if not (0.005 < wb < 0.1 and 0.01 < wbc < 0.99 and 0.2 < h < 1.0 and ok(w0, wa)): continue
            bg = Background(wb, wbc, h, ('cpl', w0, wa))
            idx.append(i); base.append(bao_abs.chi2_bg(bg) + cmb.chi2_bg(bg))
            Es.append(lambda z, bg=bg: np.interp(z, Z_LOW, bg.H_low) / (100 * bg.h))
        if idx: out[idx] = -0.5 * (np.array(base) + L.sn_chi2_batch(S, Es))
        return out

ndim, nw = len(names), 32
ck = f'results/ckpt_{case}.npz'
if os.path.exists(ck):
    z = np.load(ck); chain, lnp_hist = list(z['chain']), list(z['lnp']); state = z['chain'][-1]
else:
    chain, lnp_hist = [], []
    state = np.array(p0) * (1 + 1e-3 * np.random.default_rng(42).standard_normal((nw, ndim)))
sampler = emcee.EnsembleSampler(nw, ndim, lnp_batch, vectorize=True)
while len(chain) < NTOT and time.time() - T0 < TIME_LIMIT:
    sampler.reset()
    st = sampler.run_mcmc(state, 100, progress=False); state = st.coords
    chain += list(sampler.get_chain()); lnp_hist += list(sampler.get_log_prob())
    np.savez(ck, chain=np.array(chain), lnp=np.array(lnp_hist))
print(f"{case}: {len(chain)}/{NTOT} ステップ, {time.time()-T0:.0f}s")
if len(chain) >= NTOT:
    C = np.array(chain); LP = np.array(lnp_hist)
    tau = emcee.autocorr.integrated_time(C, tol=0); burn = int(max(500, 3 * tau.max()))
    ch = C[burn:].reshape(-1, ndim); lp = LP[burn:].ravel()
    out = {'case': case, 'nsteps': len(C), 'burn': burn, 'tau_max': float(tau.max()), 'n_eff': float(len(ch)/tau.max())}
    ext = {n: ch[:, i] for i, n in enumerate(names)}
    if 'wbc' in names:
        from early import OMEGA_NU0
        ext['Om'] = (ch[:, 1] + OMEGA_NU0) / ch[:, 2]**2; ext['H0'] = 100 * ch[:, 2]
    for k, v in ext.items():
        lo, hi = np.percentile(v, [16, 84]); out[k] = [float(v.mean()), float(v.std()), float(lo), float(hi)]
    out['chi2_min'] = float(-2 * lp.max())
    json.dump(out, open(f'results/phase1_{case}.json', 'w'), indent=1)
    print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in out.items()}, ensure_ascii=False))
