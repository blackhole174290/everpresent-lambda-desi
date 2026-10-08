"""初期宇宙の量：音響地平線 r_s(z*)、角度スケール θ*、ドラッグ期の r_d。
物理密度 ω_i = Ω_i h² を基本変数にする（CMB 圧縮尤度がこの変数で与えられるため）。
膨張率 H(z) は後期（暗黒エネルギー）部分を差し替えられるように、関数 E_late を受け取る。"""
import numpy as np
from scipy.integrate import quad

C_KMS = 299792.458
T_CMB = 2.7255
OMEGA_G = 2.47282e-5 * (T_CMB / 2.7255)**4       # 光子
NEFF = 3.044
# ニュートリノ：質量和 0.06 eV を1種が担い（CAMB の既定と同じ）、残り2種は質量ゼロ
M_NU = 0.06
TNU0_EV = 8.617333e-5 * T_CMB * (4 / 11)**(1 / 3)  # 現在のニュートリノ温度 [eV]
_FD = 7 / 8 * (4 / 11)**(4 / 3) * OMEGA_G * NEFF / 3  # 質量ゼロ1種ぶんの ω（a=1 換算）

def _F(y):
    """フェルミ分布のエネルギー積分を y=0 の値で規格化したもの（相対論的=1）"""
    if y < 1e-4: return 1.0
    f = lambda x: x * x * np.sqrt(x * x + y * y) * np.exp(-x) / (1 + np.exp(-x))
    return quad(f, 0, 50 + y)[0] / (7 * np.pi**4 / 120)

_ygrid = np.logspace(-4, 5, 400)
_Fgrid = np.array([_F(y) for y in _ygrid])
def omega_nu_massive(a):
    """質量を持つニュートリノ1種の ω(a)·a^0（=ρ/ρc,0·h²）"""
    y = M_NU * a / TNU0_EV
    F = np.interp(np.log(y), np.log(_ygrid), _Fgrid) if y > 1e-4 else 1.0
    if y > 1e5: F = y / (7 * np.pi**4 / 120) * (3 * 1.5 * 1.2020569 * 2 / 2)  # 非相対論極限（使われない安全策）
    return _FD * F / a**4

OMEGA_NU0 = omega_nu_massive(1.0)                  # ≈ 0.06/93.1

def H_of_z(z, wb, wbc, h, de_frac=None):
    """H(z) [km/s/Mpc]。平坦。de_frac(z)=ρ_DE(z)/ρ_DE(0)。既定は宇宙定数"""
    a = 1 / (1 + z)
    w_de0 = h * h - wbc - OMEGA_G - 2 * _FD - OMEGA_NU0
    rad = (OMEGA_G + 2 * _FD) / a**4 + omega_nu_massive(a)
    de = w_de0 * (de_frac(z) if de_frac else 1.0)
    return 100 * np.sqrt(wbc / a**3 + rad + de)

def z_star_HS(wb, wm):
    """Hu & Sugiyama (1996) の最終散乱赤方偏移の近似式"""
    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    return 1048 * (1 + 0.00124 * wb**-0.738) * (1 + g1 * wm**g2)

def sound_horizon(zs, wb, wbc, h, de_frac=None):
    """r_s(zs) = ∫_zs^∞ c_s/H dz を ln(1+z) で積分 [Mpc]"""
    R0 = 3 * wb / (4 * OMEGA_G)
    def f(lnx):
        z = np.exp(lnx) - 1
        cs = C_KMS / np.sqrt(3 * (1 + R0 / (1 + z)))
        return cs / H_of_z(z, wb, wbc, h, de_frac) * (1 + z)
    return quad(f, np.log(1 + zs), np.log(1 + 1e8), limit=200, epsrel=1e-9)[0]

def comoving_distance(zs, wb, wbc, h, de_frac=None):
    f = lambda z: C_KMS / H_of_z(z, wb, wbc, h, de_frac)
    return quad(f, 0, zs, limit=400, epsrel=1e-8)[0]

def theta_star(wb, wbc, h, de_frac=None, zstar=None):
    zs = zstar if zstar else z_star_HS(wb, wbc + OMEGA_NU0)
    return sound_horizon(zs, wb, wbc, h, de_frac) / comoving_distance(zs, wb, wbc, h, de_frac)

def r_drag(wb, wbc):
    """Aubourg et al. (2015) の近似式（CAMB と 0.02% 程度で一致するとされる）"""
    wnu = OMEGA_NU0
    return 55.154 * np.exp(-72.3 * (wnu + 0.0006)**2) / (wbc**0.25351 * wb**0.12807)
