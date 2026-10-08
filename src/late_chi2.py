"""低赤方偏移（z<2.5）の膨張史の「形」E(z)=H/H0 だけを使う尤度（CMB に依存しない判定用）。
- BAO（DESI DR2、相対値）：D/r_d = s·g(z)、s = (c/100)/(h r_d) を解析的に最適化。
- 超新星：絶対等級のオフセットを解析的に周辺化（likelihood.py と同じ）。
多数の実現をまとめて評価する。"""
import numpy as np
import likelihood as L

class LateShape:
    def __init__(self, sn_names=('U3', 'PP', 'DY5')):
        self.bao = L.DesiBAO()
        self.Ci = self.bao.icov; d = self.bao.val
        self.dCd = d @ self.Ci @ d; self.Cd = self.Ci @ d
        cls = {'PP': L.PantheonPlus, 'U3': L.Union3, 'DY5': L.DESY5}
        self.sn = {k: cls[k]() for k in sn_names}

    @staticmethod
    def fM_on(zq, a_rec, E):
        """a_rec（昇順）上の E(z)（行＝時刻、列＝実現）から、∫_0^z dz'/E を zq で評価"""
        lna = np.log(a_rec); g = 1 / (a_rec[:, None] * E)        # dz/E = dlna/(aE)
        cum = np.concatenate([np.zeros((1, E.shape[1])), np.cumsum(0.5 * np.diff(lna)[:, None] * (g[1:] + g[:-1]), 0)])
        tot = cum[-1]; chi_at = tot[None, :] - cum               # z(a) までの積分
        z_rec = 1 / a_rec - 1
        order = np.argsort(z_rec)
        zs, chis = z_rec[order], chi_at[order]
        return np.array([[np.interp(z, zs, chis[:, j]) for z in zq] for j in range(E.shape[1])]), zs, E[order]

    def chi2(self, a_rec, H):
        """H: (時刻, 実現)。各実現の χ²_BAO と χ²_SN を返す"""
        E = H / H[-1][None, :]                                   # a_rec の最後が a=1
        zb = self.bao.z
        fM, zs, Es = self.fM_on(zb, a_rec, E)
        Einv_b = np.array([np.interp(zb, zs, 1 / Es[:, j]) for j in range(E.shape[1])])
        G = np.empty_like(fM)
        for i, k in enumerate(self.bao.kind):
            G[:, i] = {'DM_over_rs': fM[:, i], 'DH_over_rs': Einv_b[:, i],
                       'DV_over_rs': np.cbrt(zb[i] * fM[:, i]**2 * Einv_b[:, i])}[k]
        gCd = G @ self.Cd; gCg = np.einsum('ij,jk,ik->i', G, self.Ci, G)
        out = {'BAO': self.dCd - gCd**2 / gCg}
        Elist = [(lambda z, j=j: np.interp(z, zs, Es[:, j])) for j in range(E.shape[1])]
        for k, S in self.sn.items():
            out[k] = L.sn_chi2_batch(S, Elist)
        return out
