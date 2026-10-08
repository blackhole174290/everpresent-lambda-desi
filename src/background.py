"""ベクトル化した背景膨張（本番用）。物理密度 (ω_b, ω_bc, h) と暗黒エネルギーの形を入力に、
BAO 距離（Mpc）、θ*、r_d をまとめて計算する。1回あたり数ミリ秒を目標。

較正（CAMB 2.0.4 との比較、src/check_early_vs_camb.py）：
  z* = Hu–Sugiyama 近似 − 2.02（残差のばらつき 0.003%）
  r_d = Aubourg 近似 × 1.000258
注意：この較正は「再結合期の物理が標準的」である前提。揺らぐΛが再結合期に大きいと崩れる。"""
import numpy as np
from early import (C_KMS, OMEGA_G, _FD, OMEGA_NU0, TNU0_EV, M_NU, _ygrid, _Fgrid,
                   z_star_HS, r_drag)

ZSTAR_SHIFT = -2.02
RD_FACTOR = 1.000258

def omega_nu_massive_vec(a):
    y = M_NU * a / TNU0_EV
    F = np.interp(np.log(np.maximum(y, 1e-4)), np.log(_ygrid), _Fgrid)
    return _FD * F / a**4

# 積分用の格子：低 z は z で等間隔、高 z は ln(1+z) で等間隔
Z_LOW = np.linspace(0.0, 3.0, 1501)
LNX_HIGH = np.linspace(np.log(4.0), np.log(1 + 1e7), 3000)
Z_HIGH = np.exp(LNX_HIGH) - 1

def _E2_parts(z):
    a = 1 / (1 + z)
    return a, (OMEGA_G + 2 * _FD) / a**4 + omega_nu_massive_vec(a)

_A_LOW, _RAD_LOW = _E2_parts(Z_LOW)
_A_HIGH, _RAD_HIGH = _E2_parts(Z_HIGH)

def de_cpl(z, w0, wa):
    return (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))

class Background:
    """de_frac: None（宇宙定数）、('cpl', w0, wa)、または z 配列を受ける関数 f(z)=ρ_DE(z)/ρ_DE(0)"""
    def __init__(self, wb, wbc, h, de=None):
        self.wb, self.wbc, self.h = wb, wbc, h
        w_de0 = h * h - wbc - OMEGA_G - 2 * _FD - OMEGA_NU0
        def dens(z, a, rad):
            if de is None: f = 1.0
            elif isinstance(de, tuple): f = de_cpl(z, de[1], de[2])
            else: f = de(z)
            return wbc / a**3 + rad + w_de0 * f
        self.H_low = 100 * np.sqrt(dens(Z_LOW, _A_LOW, _RAD_LOW))
        self.H_high = 100 * np.sqrt(dens(Z_HIGH, _A_HIGH, _RAD_HIGH))
        # 共動距離（低 z 部分は累積台形）
        inv = C_KMS / self.H_low
        self.chi_low = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(Z_LOW))])
        g = C_KMS / self.H_high * (1 + Z_HIGH)            # dz = (1+z) dlnx
        self.chi_high = self.chi_low[-1] + np.concatenate(
            [[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(LNX_HIGH))])
        # 音響地平線の被積分関数（高 z 側から積分）
        R0 = 3 * wb / (4 * OMEGA_G)
        cs = C_KMS / np.sqrt(3 * (1 + R0 / (1 + Z_HIGH)))
        s = cs / self.H_high * (1 + Z_HIGH)
        tail = np.concatenate([np.cumsum((0.5 * (s[1:] + s[:-1]) * np.diff(LNX_HIGH))[::-1])[::-1], [0]])
        # z_top=1e7 より上は放射優勢の解析解 ∫ c/√3 dz/H = c(1+z_top)/(√3 H(z_top)) を足す（約0.013%分）
        rs_top = C_KMS / np.sqrt(3) * (1 + Z_HIGH[-1]) / self.H_high[-1]
        self.rs_high = tail + rs_top

    def D_M(self, z):
        z = np.asarray(z, float)
        return np.where(z <= 3.0, np.interp(z, Z_LOW, self.chi_low),
                        np.interp(np.log1p(z), LNX_HIGH, self.chi_high))
    def D_H(self, z):
        return C_KMS / np.interp(z, Z_LOW, self.H_low)
    def z_star(self):
        return z_star_HS(self.wb, self.wbc + OMEGA_NU0) + ZSTAR_SHIFT
    def theta_star(self):
        zs = self.z_star()
        rs = np.interp(np.log1p(zs), LNX_HIGH, self.rs_high)
        return rs / self.D_M(zs)
    def r_d(self):
        return r_drag(self.wb, self.wbc) * RD_FACTOR
    @property
    def Om(self):
        return (self.wbc + OMEGA_NU0) / self.h**2
