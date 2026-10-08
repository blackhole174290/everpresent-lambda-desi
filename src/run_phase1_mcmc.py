"""フェーズ1-c/1-d：DESI DR2 表V の再現（emcee）。
使い方: python3 src/run_phase1_mcmc.py <ケース名>"""
import numpy as np, sys, emcee, json, time
sys.path.insert(0, 'src')
from background import Background
import likelihood as L

case = sys.argv[1]
bao = L.DesiBAOAbs()
cmb, bbn, th = L.CMBCompressed(), L.BBNPrior(), L.ThetaStarPrior()
sn = {'PP': L.PantheonPlus, 'U3': L.Union3, 'DY5': L.DESY5}

def w0wa_prior(w0, wa):
    return -3 <= w0 <= 1 and -3 <= wa <= 2 and w0 + wa < 0

# 各ケース: パラメータ名、初期値、対数尤度
if case == 'lcdm_bbn_theta':
    names = ['wb', 'wbc', 'h']; p0 = [0.0222, 0.140, 0.684]
    def lnp(p):
        wb, wbc, h = p
        if not (0.005 < wb < 0.1 and 0.01 < wbc < 0.99 and 0.2 < h < 1.0): return -np.inf
        bg = Background(wb, wbc, h)
        return -0.5 * (bao.chi2_bg(bg) + bbn.chi2_bg(bg) + th.chi2_bg(bg))
elif case == 'lcdm_cmb':
    names = ['wb', 'wbc', 'h']; p0 = [0.0222, 0.1414, 0.68]
    def lnp(p):
        wb, wbc, h = p
        if not (0.005 < wb < 0.1 and 0.01 < wbc < 0.99 and 0.2 < h < 1.0): return -np.inf
        bg = Background(wb, wbc, h)
        return -0.5 * (bao.chi2_bg(bg) + cmb.chi2_bg(bg))
elif case.startswith('w0wa_cmb'):
    s = case.split('_')[2] if case.count('_') == 2 else None
    S = sn[s]() if s else None
    names = ['wb', 'wbc', 'h', 'w0', 'wa']; p0 = [0.0222, 0.1414, 0.66, -0.7, -1.0]
    def lnp(p):
        wb, wbc, h, w0, wa = p
        if not (0.005 < wb < 0.1 and 0.01 < wbc < 0.99 and 0.2 < h < 1.0 and w0wa_prior(w0, wa)): return -np.inf
        bg = Background(wb, wbc, h, ('cpl', w0, wa))
        c = bao.chi2_bg(bg) + cmb.chi2_bg(bg)
        if S: c += L.sn_chi2_bg(S, bg)
        return -0.5 * c
elif case.startswith('w0wa_sn'):
    S = sn[case.split('_')[2]]()
    base = L.DesiBAO()
    names = ['Om', 'hrd', 'w0', 'wa']; p0 = [0.31, 100.0, -0.8, -0.5]
    def lnp(p):
        Om, hrd, w0, wa = p
        if not (0.01 < Om < 0.99 and 10 < hrd < 1000 and w0wa_prior(w0, wa)): return -np.inf
        E = lambda zg: np.sqrt(Om*(1+zg)**3 + (1-Om)*(1+zg)**(3*(1+w0+wa))*np.exp(-3*wa*zg/(1+zg)))
        return -0.5 * (base.chi2(Om, hrd, w0, wa) + S.chi2(None, E_func=E))
else:
    raise SystemExit('unknown case')

ndim, nw = len(names), 32
rng = np.random.default_rng(42)
start = np.array(p0) * (1 + 1e-3 * rng.standard_normal((nw, ndim)))
sampler = emcee.EnsembleSampler(nw, ndim, lnp)
t = time.time()
nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
sampler.run_mcmc(start, nsteps, progress=False)
try: tau = sampler.get_autocorr_time(tol=0)
except Exception: tau = np.full(ndim, np.nan)
burn = int(max(500, 3 * np.nanmax(tau)))
ch = sampler.get_chain(discard=burn, flat=True)
out = {'case': case, 'nsteps': nsteps, 'burn': burn, 'tau_max': float(np.nanmax(tau)),
       'n_eff': float(len(ch) / np.nanmax(tau)), 'time_s': round(time.time() - t)}
# 派生量
if 'wbc' in names:
    from early import OMEGA_NU0
    Om = (ch[:, 1] + OMEGA_NU0) / ch[:, 2]**2; H0 = 100 * ch[:, 2]
    extra = {'Om': Om, 'H0': H0}
else:
    extra = {}
for i, n in enumerate(names): extra[n] = ch[:, i]
for k, v in extra.items():
    lo, med, hi = np.percentile(v, [16, 50, 84])
    out[k] = [float(v.mean()), float(v.std()), float(lo), float(hi)]
lp = sampler.get_log_prob(discard=burn, flat=True)
out['chi2_min'] = float(-2 * lp.max())
json.dump(out, open(f'results/phase1_{case}.json', 'w'), indent=1)
print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in out.items()}, ensure_ascii=False))
