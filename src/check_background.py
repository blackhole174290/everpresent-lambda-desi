import numpy as np, sys, time, camb
sys.path.insert(0, 'src')
from background import Background
rng = np.random.default_rng(7); errs = []
for k in range(30):
    wb = rng.uniform(0.0215, 0.0230); wbc = rng.uniform(0.137, 0.146); h = rng.uniform(0.60, 0.74)
    w0, wa = (-1, 0) if k < 15 else (rng.uniform(-1.1, -0.3), rng.uniform(-2.5, 0.3))
    p = camb.set_params(H0=100*h, ombh2=wb, omch2=wbc-wb, mnu=0.06, nnu=3.044)
    if k >= 15: p.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
    r = camb.get_background(p); d = r.get_derived_params()
    b = Background(wb, wbc, h, None if k < 15 else ('cpl', w0, wa))
    zz = np.array([0.3, 0.5, 1.0, 2.33])
    errs.append([b.theta_star()/(d['thetastar']/100)-1, b.r_d()/d['rdrag']-1,
                 np.max(np.abs(b.D_M(zz)/r.comoving_radial_distance(zz)-1)),
                 np.max(np.abs(b.D_H(zz)*r.hubble_parameter(zz)/299792.458-1))])
e = np.array(errs)*100
print("相対誤差 [%]  平均 / 最大絶対値")
for i, n in enumerate(['θ*', 'r_d', 'D_M', 'D_H']):
    print(f"  {n:4s}: {e[:,i].mean():+.4f} / {np.abs(e[:,i]).max():.4f}")
t = time.time()
for _ in range(200): Background(0.0223, 0.1414, 0.675, ('cpl', -0.8, -0.5)).theta_star()
print(f"1回あたり {(time.time()-t)/200*1000:.2f} ms")
