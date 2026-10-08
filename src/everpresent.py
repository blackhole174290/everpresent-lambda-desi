"""揺らぐ宇宙定数（Everpresent Λ）モデル1の実現生成。
式：Das, Nasiri, Yazdi I (arXiv:2304.03819) §3–4、II (arXiv:2307.13743) §2.1
  V = (4π/3) ∫ dτ' a⁴ (τ−τ')³ = (4π/3)[τ³J₀ − 3τ²J₁ + 3τJ₂ − J₃],  J_k = ∫ a⁴ τ'^k dτ' = ∫ a³ τ'^k dt'
  S_i = S_{i−1} + α √(V_i − V_{i−1}) ξ_i,  Λ_i = 8π S_i / V_i   （G=1 の単位。II の式(2)と同値）
  H² = ω_m a⁻³ + ω_r a⁻⁴ + Λ/3          （第1フリードマン方程式のみ使用）
単位：c=1、時間・長さは 1/H100。Λ ∝ 1/√V なのでプランク長は相殺し、Λ は H100² 単位で直接計算できる。

刻み方（grid）：
  'das'  : Das et al. I の既定。a_i = i/n（n=10000）を a = a_init=1e-4（z≈10⁴）から。
           それ以前の V は Λ=0 の物質＋放射で計算（S₀=0）。
  'loga' : ln a 等間隔（a_init は任意）。
各刻みの内部では Λ を一定とし、n_sub 個の小刻みで τ, J_k を積分する（Das et al. は Romberg 法）。
H² ≤ 0 になった実現は破綻として記録し、以後は計算しない。現在は a=1、H0 は結果として得られる量。"""
import numpy as np

OMEGA_R = 2.47282e-5 * (1 + 0.2271 * 3.044)

def _initial_state(wm, wr, a_init, a_start=1e-9, n=20000):
    """Λ=0 の物質＋放射宇宙で τ と J_k を a_init まで積分（a_start 以前は放射優勢の解析解）"""
    A = np.sqrt(2 * np.sqrt(wr)); t0 = a_start**2 / (2 * np.sqrt(wr))
    tau = 2 * t0 / a_start
    J = np.array([A**(3 - k) * 2**k * t0**((5 + k) / 2) / ((5 + k) / 2) for k in range(4)])
    lna = np.linspace(np.log(a_start), np.log(a_init), n)
    a = np.exp(lna); H = np.sqrt(wm / a**3 + wr / a**4)
    dtau = np.concatenate([[0], np.cumsum(0.5 * np.diff(lna) * ((1 / (a * H))[1:] + (1 / (a * H))[:-1]))])
    taus = tau + dtau
    for k in range(4):
        f = a**3 * taus**k / H
        J[k] += np.sum(0.5 * np.diff(lna) * (f[1:] + f[:-1]))
    return taus[-1], J

def _vol(tau, J):
    return 4 * np.pi / 3 * (tau**3 * J[0] - 3 * tau**2 * J[1] + 3 * tau * J[2] - J[3])

def a_grid(kind='das', n=10000, a_init=1e-4):
    if kind == 'das':
        i0 = int(round(a_init * n)); return np.arange(i0, n + 1) / n
    if kind == 'loga':
        return np.exp(np.linspace(np.log(a_init), 0.0, n + 1))
    raise ValueError(kind)

def realize(wm, alpha, n_real, seed, grid='das', n=10000, a_init=1e-4, n_sub=4, wr=OMEGA_R,
            keep_every=10, S0=0.0, volume='full', rec_amin=0.0):
    """volume='full'：開始前の領域も含めた厳密な過去光円錐の体積（ΔV も厳密）。
       volume='das' ：Das et al. I §5.3 の方式。V = V₀（開始時点の体積、定数）+ V_after(τ)。
                      V_after は開始時刻 τ_i 以後の領域のみ、ΔV も V_after の差分。"""
    rng = np.random.default_rng(seed)
    ag = a_grid(grid, n, a_init)
    tau0, J0 = _initial_state(wm, wr, ag[0])
    N = n_real
    tau = np.full(N, tau0)
    V0 = _vol(tau0, J0)
    J = np.tile((J0 if volume == 'full' else np.zeros(4))[:, None], (1, N))
    Vf = lambda: _vol(tau, J) + (0.0 if volume == 'full' else V0)
    V = Vf(); S = np.full(N, S0); Lam = 8 * np.pi * S / V
    alive = np.ones(N, bool); crash_a = np.full(N, np.nan)
    rec_a, rec_H, rec_L = [ag[0]], [np.sqrt(wm / ag[0]**3 + wr / ag[0]**4 + Lam / 3)], [Lam.copy()]
    for i in range(1, len(ag)):
        a0_, a1_ = ag[i - 1], ag[i]
        sub = np.exp(np.linspace(np.log(a0_), np.log(a1_), n_sub + 1))
        dl = np.log(a1_ / a0_) / n_sub
        # 刻みの内部で Λ は一定（Λ_{i−1}）
        h2s = wm / sub[:, None]**3 + wr / sub[:, None]**4 + Lam[None, :] / 3
        bad = alive & (h2s.min(0) <= 0)
        crash_a[bad] = a1_; alive &= ~bad
        Hs = np.sqrt(np.where(alive[None, :], h2s, 1.0))
        g = 1 / (sub[:, None] * Hs)                          # dτ/dlna
        taus = tau[None, :] + np.concatenate([np.zeros((1, N)), np.cumsum(0.5 * dl * (g[1:] + g[:-1]), 0)])
        for k in range(4):
            f = sub[:, None]**3 * taus**k / Hs                 # a³ τ^k dt/dlna
            J[k] += np.sum(0.5 * dl * (f[1:] + f[:-1]), 0)
        tau = taus[-1]
        Vn = Vf()
        S = S + alpha * np.sqrt(np.maximum(Vn - V, 0.0)) * rng.standard_normal(N)
        V = Vn
        Lam = np.where(alive, 8 * np.pi * S / V, Lam)
        h2 = wm / a1_**3 + wr / a1_**4 + Lam / 3
        bad = alive & (h2 <= 0); crash_a[bad] = a1_; alive &= ~bad
        if (i % keep_every == 0 or i == len(ag) - 1) and a1_ >= rec_amin:
            rec_a.append(a1_); rec_H.append(np.where(alive, np.sqrt(np.abs(h2)), np.nan))
            rec_L.append(np.where(alive, Lam, np.nan))
    H = rec_H[-1]
    return {'a': np.array(rec_a), 'H': np.array(rec_H), 'Lam': np.array(rec_L), 'alive': alive,
            'crash_a': crash_a, 'h': H, 'OmL0': np.where(alive, Lam / 3 / H**2, np.nan), 'V0': V,
            'wm': wm, 'alpha': alpha, 'seed': seed, 'grid': grid, 'n': n, 'a_init': a_init}
