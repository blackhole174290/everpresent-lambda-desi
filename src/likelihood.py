"""尤度。まず DESI DR2 BAO（ガウス近似、公式の平均と共分散）。"""
import numpy as np, os
from cosmology import distances_over_hrd

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'desi_dr2')

class DesiBAO:
    def __init__(self):
        rows = [l.split() for l in open(os.path.join(DATA, 'desi_gaussian_bao_ALL_GCcomb_mean.txt'))
                if not l.startswith('#')]
        self.z = np.array([float(r[0]) for r in rows])
        self.val = np.array([float(r[1]) for r in rows])
        self.kind = [r[2] for r in rows]
        cov = np.loadtxt(os.path.join(DATA, 'desi_gaussian_bao_ALL_GCcomb_cov.txt'))
        self.icov = np.linalg.inv(cov)

    def model(self, Om, hrd, w0=-1.0, wa=0.0):
        DM, DH, DV = distances_over_hrd(self.z, Om, hrd, w0, wa)
        pick = {'DM_over_rs': DM, 'DH_over_rs': DH, 'DV_over_rs': DV}
        return np.array([pick[k][i] for i, k in enumerate(self.kind)])

    def chi2(self, Om, hrd, w0=-1.0, wa=0.0):
        d = self.val - self.model(Om, hrd, w0, wa)
        return d @ self.icov @ d


# ---------------------------------------------------------------
# Ia型超新星。絶対等級（と H0）の縮退した定数オフセットを解析的に周辺化する。
#   χ²_marg = a - b²/e + ln(e/2π),  a = rᵀC⁻¹r, b = Σ(C⁻¹r), e = ΣC⁻¹
# 距離：d_L = (1+z_hel) D_M(z_cosmo)。H0 はオフセットに吸収されるので c/H0 = 1 で計算。
from cosmology import E_of_z
from scipy.integrate import cumulative_trapezoid
import pandas as pd

SNDIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sn')

def _read_cov(path, n):
    v = np.loadtxt(path, skiprows=1)
    assert int(open(path).readline()) == n and v.size == n * n
    return v.reshape(n, n)

class SupernovaBase:
    def _setup(self, zc, zh, mag, cov):
        self.zc, self.zh, self.mag = zc, zh, mag
        self.icov = np.linalg.inv(cov)
        self.e = self.icov.sum()
        self.zmax = zc.max() * 1.01
        print(f"  {self.name}: {len(zc)} 点, z = {zc.min():.3f}–{zc.max():.3f}")

    def mu_shape(self, Om, w0=-1.0, wa=0.0, E_func=None, n=4000):
        """5log10[(1+z_hel) D_M H0/c]。E_func を渡せば任意の H(z)/H0 に対応（後で揺らぐΛに使う）"""
        zg = np.linspace(0, self.zmax, n)
        E = E_func(zg) if E_func is not None else E_of_z(zg, Om, w0, wa)
        chi = cumulative_trapezoid(1 / E, zg, initial=0.0)
        return 5 * np.log10((1 + self.zh) * np.interp(self.zc, zg, chi))

    def chi2(self, Om, w0=-1.0, wa=0.0, E_func=None):
        r = self.mag - self.mu_shape(Om, w0, wa, E_func)
        Cr = self.icov @ r
        a, b = r @ Cr, Cr.sum()
        return a - b * b / self.e + np.log(self.e / (2 * np.pi))

class PantheonPlus(SupernovaBase):
    name = 'Pantheon+'
    def __init__(self):
        d = pd.read_csv(os.path.join(SNDIR, 'PantheonPlus', 'Pantheon+SH0ES.dat'), sep=r'\s+')
        cov = _read_cov(os.path.join(SNDIR, 'PantheonPlus', 'Pantheon+SH0ES_STAT+SYS.cov'), len(d))
        m = (d.zHD > 0.01).values  # DESI・Cobaya と同じカット（SH0ES 較正は使わない）
        self._setup(d.zHD.values[m], d.zHEL.values[m], d.m_b_corr.values[m], cov[np.ix_(m, m)])

class DESY5(SupernovaBase):
    name = 'DES-SN5YR'
    def __init__(self):
        d = pd.read_csv(os.path.join(SNDIR, 'DESY5', 'DES-SN5YR_HD.csv'))
        cov = _read_cov(os.path.join(SNDIR, 'DESY5', 'covsys_000.txt'), len(d))
        cov += np.diag(d.MUERR_FINAL.values**2)  # 系統誤差＋統計誤差（対角）
        self._setup(d.zHD.values, d.zHEL.values, d.MU.values, cov)

class Union3(SupernovaBase):
    name = 'Union3'
    def __init__(self):
        d = pd.read_csv(os.path.join(SNDIR, 'Union3', 'lcparam_full.txt'), sep=r'\s+')
        cov = _read_cov(os.path.join(SNDIR, 'Union3', 'mag_covmat.txt'), len(d))
        self._setup(d.zcmb.values, d.zhel.values, d.mb.values, cov)  # 22 ビンの圧縮データ


# ---------------------------------------------------------------
# 絶対スケールの尤度（Background オブジェクトを受け取る）
class DesiBAOAbs(DesiBAO):
    """BAO を r_d で割った距離として、Background から直接評価（CMB/BBN と組むとき用）"""
    def chi2_bg(self, bg):
        rd = bg.r_d()
        DM = bg.D_M(self.z) / rd; DH = bg.D_H(self.z) / rd
        DV = np.cbrt(self.z * DM**2 * DH)
        pick = {'DM_over_rs': DM, 'DH_over_rs': DH, 'DV_over_rs': DV}
        m = np.array([pick[k][i] for i, k in enumerate(self.kind)])
        d = self.val - m
        return d @ self.icov @ d

class CMBCompressed:
    """(θ*, ω_b, ω_bc) のガウス近似。Planck PR4 CamSpec 早期宇宙チェーン（Lemos & Lewis 2023,
    github.com/cmbant/PlanckEarlyLCDM）から getdist で計算した値。DESI DR2 付録 A と同じ共分散。
    ω_bc = ω_b + ω_c（ニュートリノを含まない）。DESI 論文の 0.14208 は ω_ν=0.00064 込みの値"""
    mean = np.array([1.04102739e-02, 0.0222321, 0.1414339])
    cov = 1e-9 * np.array([[6.620994e-03, 1.244421e-01, -1.192875e+00],
                           [1.244421e-01, 2.134417e+01, -9.400084e+01],
                           [-1.192875e+00, -9.400084e+01, 1.488417e+03]])
    def __init__(self): self.icov = np.linalg.inv(self.cov)
    def chi2_bg(self, bg):
        d = np.array([bg.theta_star(), bg.wb, bg.wbc]) - self.mean
        return d @ self.icov @ d

class BBNPrior:
    """DESI DR2 の BBN 事前分布 ω_b = 0.02218 ± 0.00055"""
    def chi2_bg(self, bg): return ((bg.wb - 0.02218) / 0.00055)**2

class ThetaStarPrior:
    """DESI DR2 の θ* 事前分布 100θ* = 1.04110 ± 0.00053"""
    def chi2_bg(self, bg): return ((100 * bg.theta_star() - 1.04110) / 0.00053)**2

def sn_chi2_bg(sn, bg):
    from background import Z_LOW
    E = lambda zg: np.interp(zg, Z_LOW, bg.H_low) / (100 * bg.h)
    return sn.chi2(None, E_func=E)


def sn_chi2_batch(sn, E_list, n=3000):
    """複数モデルの SN χ² をまとめて計算（行列×行列で高速化）。E_list: E(z) 配列を返す関数のリスト"""
    zg = np.linspace(0, sn.zmax, n)
    M = np.empty((len(E_list), len(sn.zc)))
    for i, E in enumerate(E_list):
        inv = 1 / E(zg)
        chi = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zg))])
        M[i] = 5 * np.log10((1 + sn.zh) * np.interp(sn.zc, zg, chi))
    R = sn.mag[None, :] - M
    CR = R @ sn.icov
    a = np.einsum('ij,ij->i', R, CR); b = CR.sum(1)
    return a - b * b / sn.e + np.log(sn.e / (2 * np.pi))


class DESDovekie(SupernovaBase):
    """DES-SN5YR の再較正版（DES-Dovekie）。共分散は「逆行列の上三角を詰めたもの」で配布されている"""
    name = 'DES-Dovekie'
    def __init__(self):
        d = pd.read_csv(os.path.join(SNDIR, 'DES-Dovekie', 'DES-Dovekie_HD.csv'))
        z = np.load(os.path.join(SNDIR, 'DES-Dovekie', 'covtot_inv_000.npz'))
        n = len(d); ic = np.zeros((n, n)); ic[np.triu_indices(n)] = z['cov']
        ic = ic + ic.T - np.diag(np.diag(ic))
        self.zc, self.zh, self.mag = d.zHD.values, d.zHEL.values, d.MU.values
        self.icov = ic; self.e = ic.sum(); self.zmax = self.zc.max() * 1.01
        print(f"  {self.name}: {n} 点, z = {self.zc.min():.3f}–{self.zc.max():.3f}")
