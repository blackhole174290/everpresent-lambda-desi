"""背景膨張：H(z) と距離。ΛCDM と w0waCDM（平坦）。
単位：距離は Mpc、c は km/s。"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

C_KMS = 299792.458
# 放射（光子＋ニュートリノ、N_eff=3.044 相当）の Ω_r h^2。BAO 範囲 z<2.4 では影響は微小
OMEGA_R_H2 = 2.469e-5 * (1 + 0.2271 * 3.044)

def E_of_z(z, Om, w0=-1.0, wa=0.0, h=0.68):
    """E(z)=H(z)/H0。平坦。w(a)=w0+wa(1-a)"""
    z = np.asarray(z, float)
    Or = OMEGA_R_H2 / h**2
    Ode = 1.0 - Om - Or
    a = 1.0 / (1.0 + z)
    de = a**(-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a))
    return np.sqrt(Om * (1 + z)**3 + Or * (1 + z)**4 + Ode * de)

def distances_over_hrd(zs, Om, hrd, w0=-1.0, wa=0.0, zmax=3.0, n=6000):
    """D_M/r_d, D_H/r_d, D_V/r_d を返す。hrd = h * r_d [Mpc]。
    D_X = (c/H0) f(z) = (c/100)/h * f  →  D_X/r_d = (c/100) f / (h r_d)"""
    zg = np.linspace(0, zmax, n)
    inv = 1.0 / E_of_z(zg, Om, w0, wa)
    chi = cumulative_trapezoid(inv, zg, initial=0.0)  # 無次元の共動距離
    fac = (C_KMS / 100.0) / hrd
    zs = np.asarray(zs)
    DM = fac * np.interp(zs, zg, chi)
    DH = fac / E_of_z(zs, Om, w0, wa)
    DV = np.cbrt(zs * DM**2 * DH)
    return DM, DH, DV
