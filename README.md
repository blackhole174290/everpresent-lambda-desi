# 揺らぐ宇宙定数（Everpresent Λ、モデル1）を DESI DR2 で検証する

Testing Everpresent Λ (Model 1) with DESI DR2 BAO, supernovae and CMB — *English summary below.*

ソーキンの「揺らぐ宇宙定数」の現象論的モデル（Ahmed et al. 2004 のモデル1、Das et al. 2023 の実装）を、公開データ（DESI DR2 BAO、超新星4種、CMB 圧縮尤度）と比較したコードと結果です。すべての結果は、このリポジトリのスクリプトと公開データから再生成できます。

## 結果の要約

| 判定に使ったデータ | 生存率（全実現のうち Δχ² < 9 の割合） |
| --- | --- |
| DESI DR2 BAO + CMB | 0（約1万2千実現、最良でも Δχ² = +466） |
| DESI DR2 BAO 単独 | 約 1×10⁻⁵（α = 0.007〜0.009 の30万実現中4個） |
| DESI DR2 BAO + Pantheon+ | 0（95% 上限 5.5×10⁻⁶、計55万実現） |
| DESI DR2 BAO + Union3 / DES-SN5YR | 約 2×10⁻⁶（55万実現中1個） |

- Δχ² は、各実現の χ² と ΛCDM の最良 χ² の差。設定は S₀ = 0、z = 10⁴ 開始、ln a 等間隔2000分割（頑健性として8000分割も確認）。
- BAO 単独に合う実現は超新星で大きく外れる。揺らぐΛはおよそハッブル時間ごとにうねるため、BAO（z = 0.3〜2.3）と超新星（z < 1）が別々のうねり方を要求する。
- 唯一 BAO + 超新星を通過した実現（seed 1558005）は、現在の Ω_m ≈ 0.58、CMB の ω_m を当てると H₀ ≈ 50 km/s/Mpc、θ* が 6.6% ずれる宇宙で、他の観測と両立しない。
- これはモデル1という特定の実装についての結論で、揺らぐ宇宙定数のアイデア全般を否定するものではない。

## 検証の過程で気づいた2点

**A. DESI DR2 付録Aの CMB 圧縮尤度の第3成分**
付録Aの式(A1)で ω_bc とされる値 0.14208 は、元のチェーン（cmbant/PlanckEarlyLCDM）のニュートリノ込みの派生量 omegamh2 の平均 0.14207901 と一致します。チェーンの ombh2 + omch2 の平均は 0.141434 です。DESI DR2 論文の第I節では ω_bc はニュートリノを含まない量として定義されているので、両者には 0.00064（ω_bc の誤差の約1.7σ）の差があります。モデルの ω_b + ω_c を 0.14208 と比べると、DESI の表VIの Δχ² = −8.0 が再現できます（本コードで −8.10、定義どおりでは −6.66）。DESI の主要な結論（フル CMB 尤度）には影響しません。これは数値の一致にもとづく観察で、DESI 側の実装は確認していません。

**B. モデル1の完走率は計算の刻みに依存する**
途中で H² < 0 にならない実現の割合（完走率）は、刻みを細かくするほど √刻み幅 に比例して下がり、連続極限では 0.174 ± 0.005 です（α = 0.01、ω_m h² = 0.14）。Das et al. (2023) の 0.323 は特定の刻みでの値と考えられます。生存した実現の H₀ 分布などの統計は刻みにほぼ依存しません。

## 再現の手順

Python 3.12 で動作確認しています。

```
pip install -r requirements.txt
python fetch_data.py
```

| 段階 | コマンド | 内容 | 目安の時間（1コア） |
| --- | --- | --- | --- |
| 既知結果の再現 | python src/run_lcdm_bao.py | DESI BAO 単独の ΛCDM | 数秒 |
| | python src/run_lcdm_sn.py | 超新星単独の ΛCDM | 数十秒 |
| | python src/check_background.py | 初期宇宙の量を CAMB と比較 | 数十秒 |
| | python src/run_phase1_mcmc.py w0wa_cmb 5000 | DESI+CMB 圧縮の w0wa | 約1分 |
| | python src/run_phase1_mcmc_v2.py w0wa_sn_DY5 4000 | DESI+超新星の w0wa（PP / U3 / DY5） | 数分 |
| | python src/run_dchi2.py | Δχ²（w0wa − ΛCDM）の最適化 | 数分 |
| モデル1の再現 | python src/run_phase2_das_repro.py 3000 full | Das et al. の統計との比較 | 約30秒 |
| | python src/run_phase2_stepconv.py 0.01 4000 2000 4000 8000 | 完走率の刻み依存（発見B） | 数分 |
| 主結果 | python src/run_phase3_late.py 20000 | 6つの α で各2万実現 | 約2分 |
| | python src/run_phase3_late.py 100000 0.007 0.008 0.009 | 最良領域を各10万実現 | 約6分 |
| | python src/summarize_phase3_late.py | 生存率の集計 | 約1分 |
| 頑健性 | NSTEP=8000 python src/run_phase3_late.py 50000 0.008 0.009 | 8000分割での再計算 | 約3分 |
| | python src/compare_stepsize.py | 刻み幅の比較 | 数秒 |

- Windows では環境変数を `set NSTEP=8000`（コマンドプロンプト）または `$env:NSTEP=8000`（PowerShell）で設定してから実行します。
- 実行時間に上限がある環境では `TIME_LIMIT`（秒）を設定すると、途中で保存して終了し、再実行で続きから計算します。
- 乱数の種は α・バッチ番号・刻み数から一意に決まるので、同じ結果が再生成されます。

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| src/background.py, src/early.py | 背景膨張、θ*、r_d（CAMB 2.0.4 で較正） |
| src/cosmology.py | 相対 BAO 用の簡易な距離計算 |
| src/likelihood.py | DESI BAO、超新星4種、CMB 圧縮尤度 |
| src/everpresent.py | モデル1の実現生成（体積を O(1) で更新、多数の実現をベクトル化） |
| src/everpresent_bg.py | 実現の膨張史を背景膨張モジュールに接続 |
| src/late_chi2.py | CMB を使わない判定用の尤度（多数の実現を一括評価） |
| notes/phase0.md〜phase4.md | 各段階の判定記録（日本語） |
| results/ | 集計結果（JSON）と唯一の生存実現の膨張史 |

## 注意点

- 結論はモデル1の特定の設定（S₀ = 0、z = 10⁴ 開始）と、背景膨張だけの解析に限られます。Zuntz (2008) が指摘した空間的な揺らぎによる CMB の制約は扱っていません。
- 再結合期の Λ が大きい実現では、CMB 圧縮尤度の前提（標準的な初期宇宙）が成り立ちません。CMB 込みの結論は、ずれが桁違いに大きいことに依拠しています。
- 著者は宇宙論の専門家ではなく、独学で取り組んだ研究です。誤りの指摘を歓迎します。

## AI の利用について

本研究は、Anthropic の Claude（Claude Opus 5.5）の支援を受けて行いました。コードの作成、計算の実行、文献の調査、文章の作成に AI を使用しています。研究の方向づけ、各段階の判断、公表の決定は著者が行い、内容の責任は著者が負います。

結果の信頼性は、AI の出力をそのまま受け入れるのではなく、既存の公表値（DESI DR2、Das et al. 2023）の再現と、CAMB との数値比較で確かめる方針をとりました。検証の過程で見つかった誤りと修正は notes/ に記録しています。

## 引用

CITATION.cff を参照してください。データを使う場合は、data/README.md に挙げた各論文も引用してください。

## ライセンス

コードは MIT License です。データはこのリポジトリに含まれず、それぞれの提供元の条件に従います。

---

## English summary

This repository tests the phenomenological "Everpresent Λ" model (Model 1 of Ahmed et al. 2004, as implemented by Das et al. 2023) against public data: DESI DR2 BAO, four supernova samples (Pantheon+, DES-SN5YR, DES-Dovekie, Union3) and a compressed CMB likelihood. All results can be regenerated from the scripts here.

**Main result.** With CMB, no realization comes close to ΛCDM (best Δχ² = +466 among ~12k realizations). Without CMB, about 1 in 10⁵ realizations fits DESI DR2 BAO alone (Δχ² < 9), but only 1 in 550k fits BAO plus supernovae, and that one has Ω_m ≈ 0.58 and H₀ ≈ 50 km/s/Mpc. The stochastic Λ oscillates on roughly a Hubble time, so BAO (z = 0.3–2.3) and supernovae (z < 1) require different realizations. This concerns Model 1 specifically, not the idea of a fluctuating cosmological constant in general.

**Observation A.** The third component of the compressed CMB mean vector in Appendix A of DESI DR2 (Eq. A1), 0.14208, matches the mean of the derived parameter omegamh2 (including massive neutrinos) in the cmbant/PlanckEarlyLCDM chains (0.14207901), while ombh2 + omch2 averages to 0.141434. The paper defines ω_bc without neutrinos, so the difference is 0.00064 (about 1.7σ in ω_bc). Comparing model ω_b + ω_c with 0.14208 reproduces DESI's Δχ² = −8.0 (−8.10 here; −6.66 with 0.141434). This does not affect DESI's main results, which use the full CMB likelihood. It is an observation based on matching numbers; we have not inspected DESI's implementation.

**Observation B.** The fraction of Model 1 realizations that avoid H² < 0 depends on the time step, decreasing as √(step size); the continuum limit is 0.174 ± 0.005 (α = 0.01, ω_m h² = 0.14), versus 0.323 reported by Das et al. Survivor statistics such as the H₀ distribution are insensitive to the step.

**Use of AI.** This work was carried out with the assistance of Anthropic's Claude (Claude Opus 5.5), which was used for writing code, running computations, surveying the literature and drafting text. The author directed the research, made the decisions at each stage and the decision to publish, and takes responsibility for the content. Reliability was checked by reproducing published results (DESI DR2, Das et al. 2023) and by numerical comparison with CAMB, rather than by accepting AI output as is; errors found and corrected along the way are recorded in notes/.

The author is an independent, self-taught researcher rather than a professional cosmologist. Corrections are welcome. Code: MIT License.
