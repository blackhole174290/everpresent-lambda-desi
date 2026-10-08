"""自作の初期宇宙計算を CAMB と比較する。ΛCDM と w0wa の両方で。"""
import numpy as np, sys, camb
sys.path.insert(0, 'src')
import early
rng = np.random.default_rng(1)
rows = []
for k in range(40):
    wb = rng.uniform(0.0215, 0.0230); wbc = rng.uniform(0.137, 0.146)
    h = rng.uniform(0.62, 0.72)
    w0, wa = (-1, 0) if k < 20 else (rng.uniform(-1.1, -0.4), rng.uniform(-2, 0.3))
    p = camb.set_params(H0=100 * h, ombh2=wb, omch2=wbc - wb, mnu=0.06, nnu=3.044, tau=0.055)
    if k >= 20:
        p.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
    d = camb.get_background(p).get_derived_params()
    fr = None if k < 20 else (lambda z, w0=w0, wa=wa: (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z)))
    zs_hs = early.z_star_HS(wb, wbc + early.OMEGA_NU0)
    th_hs = early.theta_star(wb, wbc, h, fr)
    th_cz = early.theta_star(wb, wbc, h, fr, zstar=d['zstar'])   # z* だけ CAMB を借りた場合
    rd = early.r_drag(wb, wbc)
    rows.append([zs_hs - d['zstar'], th_hs / (d['thetastar'] / 100) - 1, th_cz / (d['thetastar'] / 100) - 1,
                 rd / d['rdrag'] - 1])
r = np.array(rows)
sig_th = np.sqrt(6.620994e-12) / 0.0104103  # 圧縮尤度での θ* の相対誤差
print(f"θ* の許容誤差（圧縮尤度の1σ）: {sig_th*100:.4f}%")
for name, sl in [('ΛCDM', slice(0, 20)), ('w0wa', slice(20, 40))]:
    x = r[sl]
    print(f"[{name}] z*(HS)-z*(CAMB): 平均 {x[:,0].mean():+.3f}, 幅 {x[:,0].std():.3f}")
    print(f"        θ*(HS)/CAMB-1   : 平均 {x[:,1].mean()*100:+.4f}%, 幅 {x[:,1].std()*100:.4f}%")
    print(f"        θ*(z*=CAMB)/CAMB-1: 平均 {x[:,2].mean()*100:+.4f}%, 幅 {x[:,2].std()*100:.4f}%")
    print(f"        r_d(Aubourg)/CAMB-1: 平均 {x[:,3].mean()*100:+.4f}%, 幅 {x[:,3].std()*100:.4f}%")
