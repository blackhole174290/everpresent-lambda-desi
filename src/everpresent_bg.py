"""揺らぐΛの実現を、フェーズ1の Background（BAO 距離・θ*・r_d）に渡すための接続部。
- 暗黒エネルギーを「物理密度 ω_Λ(z) = Λ(z)/3（H100² 単位）」の表で与える。
- h は自由パラメータではなく、h² = ω_bc + ω_r + ω_ν + ω_Λ(0) から決まる（モデル1では H0 は結果）。
- z > z_init（実現の開始より前）では ω_Λ = 0。
注意（フェーズ3で対処）：
  1) r_d は Aubourg 近似（暗黒エネルギーを含まない）のまま。再結合期の ω_Λ が無視できない実現では不正確。
  2) モデル1側は ν を質量ゼロとして扱い、Background 側は 0.06 eV を入れているので、ω_m の定義に約0.5%の不整合がある。"""
import numpy as np
import background as B
from early import OMEGA_G, _FD, OMEGA_NU0

class EverpresentBackground(B.Background):
    def __init__(self, wb, wbc, a_tab, omL_tab):
        self.wb, self.wbc = wb, wbc
        lnx_tab = -np.log(a_tab)                        # ln(1+z)
        order = np.argsort(lnx_tab)
        lx, om = lnx_tab[order], omL_tab[order]
        omL = lambda z: np.interp(np.log1p(z), lx, om, right=0.0)
        self.h = float(np.sqrt(wbc + OMEGA_G + 2 * _FD + OMEGA_NU0 + omL(0.0)))
        def H(z, a, rad):
            return 100 * np.sqrt(wbc / a**3 + rad + omL(z))
        self.H_low = H(B.Z_LOW, B._A_LOW, B._RAD_LOW)
        self.H_high = H(B.Z_HIGH, B._A_HIGH, B._RAD_HIGH)
        inv = B.C_KMS / self.H_low
        self.chi_low = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(B.Z_LOW))])
        g = B.C_KMS / self.H_high * (1 + B.Z_HIGH)
        self.chi_high = self.chi_low[-1] + np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(B.LNX_HIGH))])
        R0 = 3 * wb / (4 * OMEGA_G)
        cs = B.C_KMS / np.sqrt(3 * (1 + R0 / (1 + B.Z_HIGH)))
        s = cs / self.H_high * (1 + B.Z_HIGH)
        tail = np.concatenate([np.cumsum((0.5 * (s[1:] + s[:-1]) * np.diff(B.LNX_HIGH))[::-1])[::-1], [0]])
        self.rs_high = tail + B.C_KMS / np.sqrt(3) * (1 + B.Z_HIGH[-1]) / self.H_high[-1]
        self.omL_rec = float(omL(1090.0) / (wbc * 1091**3 + (OMEGA_G + 3 * _FD) * 1091**4))  # 再結合期の Λ の割合


def z_drag_EH(wb, wm):
    """Eisenstein & Hu (1998) のドラッグ期の近似式"""
    b1 = 0.313 * wm**-0.419 * (1 + 0.607 * wm**0.674); b2 = 0.238 * wm**0.223
    return 1291 * wm**0.251 / (1 + 0.659 * wm**0.828) * (1 + b1 * wb**b2)

_rd_ratio_cache = {}
def _rd_ratio(wb, wbc):
    """標準的な初期宇宙で「較正済み Aubourg 近似 / (EH の z_d での積分)」の比。z_d 近似の誤差を吸収する"""
    key = (round(wb, 7), round(wbc, 7))
    if key not in _rd_ratio_cache:
        std = B.Background(wb, wbc, 0.68)
        zd = z_drag_EH(wb, wbc + OMEGA_NU0)
        rs = np.interp(np.log1p(zd), B.LNX_HIGH, std.rs_high)
        _rd_ratio_cache[key] = std.r_d() / rs
    return _rd_ratio_cache[key]

def _r_d_with_lambda(self):
    """r_d を、この実現の H(z)（再結合期の Λ を含む）で積分し、標準ケースでの較正比を掛ける。
    注意：z_d 自体の Λ による変化（再結合の時期のずれ）は無視している。"""
    zd = z_drag_EH(self.wb, self.wbc + OMEGA_NU0)
    return float(np.interp(np.log1p(zd), B.LNX_HIGH, self.rs_high)) * _rd_ratio(self.wb, self.wbc)

EverpresentBackground.r_d = _r_d_with_lambda
