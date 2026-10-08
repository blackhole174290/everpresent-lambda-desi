"""ΔΧ²_MAP（w0waCDM − ΛCDM）を最適化で求め、DESI DR2 表VI と比較する。
事前分布：w0∈[-3,1], wa∈[-3,2], w0+wa<0（範囲外は大きなペナルティ）。複数の初期点から Nelder–Mead。"""
import numpy as np, sys, json
from scipy.optimize import minimize
sys.path.insert(0, 'src')
from background import Background
import likelihood as L
bao, baoA, cmb = L.DesiBAO(), L.DesiBAOAbs(), L.CMBCompressed()
SN = {k: c() for k, c in [('PP', L.PantheonPlus), ('U3', L.Union3), ('DY5', L.DESY5)]}
BIG = 1e10
def ok(w0, wa): return -3 <= w0 <= 1 and -3 <= wa <= 2 and w0 + wa < 0
def Erel(Om, w0, wa): return lambda z: np.sqrt(Om*(1+z)**3 + (1-Om)*(1+z)**(3*(1+w0+wa))*np.exp(-3*wa*z/(1+z)))

def make(case, dyn):
    if case == 'bao' or case.startswith('sn_'):
        S = SN[case[3:]] if case.startswith('sn_') else None
        def f(p):
            Om, hrd, w0, wa = (*p, -1.0, 0.0) if not dyn else p
            if not (0.05 < Om < 0.9 and ok(w0, wa)): return BIG
            c = bao.chi2(Om, hrd, w0, wa)
            if S: c += S.chi2(None, E_func=Erel(Om, w0, wa))
            return c
        starts = [[0.30, 101.0]] if not dyn else [[0.30, 101, -0.9, -0.3], [0.33, 98, -0.7, -1.0], [0.35, 97, -0.45, -1.8], [0.36, 96, -0.3, -2.6]]
    else:  # cmb
        def f(p):
            wb, wbc, h, w0, wa = (*p, -1.0, 0.0) if not dyn else p
            if not (0.3 < h < 1.0 and ok(w0, wa)): return BIG
            bg = Background(wb, wbc, h, None if not dyn else ('cpl', w0, wa))
            return baoA.chi2_bg(bg) + cmb.chi2_bg(bg)
        starts = [[0.0223, 0.1400, 0.684]] if not dyn else [[0.0222, 0.1414, 0.64, -0.45, -1.7], [0.0222, 0.1414, 0.62, -0.3, -2.4], [0.0222, 0.1414, 0.66, -0.7, -1.0], [0.0222, 0.1414, 0.60, -0.1, -2.9]]
    return f, starts

def best(f, starts):
    res = []
    for s in starts:
        r = minimize(f, s, method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-6, 'maxiter': 20000, 'maxfev': 20000, 'adaptive': True})
        r = minimize(f, r.x, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-7, 'maxiter': 20000, 'maxfev': 20000, 'adaptive': True})
        res.append(r)
    return min(res, key=lambda r: r.fun)

desi = {'bao': -4.7, 'cmb': -8.0, 'sn_PP': -4.9, 'sn_U3': -10.1, 'sn_DY5': -13.6}
out = {}
for case in ([] if __name__ != "__main__" else (sys.argv[1:] or desi)):
    fL, sL = make(case, False); fW, sW = make(case, True)
    rL, rW = best(fL, sL), best(fW, sW)
    d = rW.fun - rL.fun
    out[case] = {'chi2_LCDM': rL.fun, 'chi2_w0wa': rW.fun, 'dchi2': d, 'best_w0wa': rW.x.tolist(), 'desi_dchi2': desi[case]}
    print(f"{case:7s}: Δχ² = {d:+.2f}  (DESI {desi[case]:+.1f})   w0wa 最良点 {np.round(rW.x, 3)}")
    json.dump(out, open(f'results/dchi2_{"_".join(sys.argv[1:]) or "all"}.json', 'w'), indent=1)
