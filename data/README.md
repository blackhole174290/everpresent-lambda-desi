# データ

データはこのリポジトリに含めていない。リポジトリ直下で次を実行すると、取得元のコミットを固定して data/ に置かれる（git は不要）。

```
python fetch_data.py
```

| 置き先 | 内容 | 取得元（コミット固定） |
| --- | --- | --- |
| data/desi_dr2/ | DESI DR2 BAO 13点と共分散（desi_gaussian_bao_ALL_GCcomb_*） | github.com/CobayaSampler/bao_data @ bb0c1c9 |
| data/sn/PantheonPlus/ | Pantheon+SH0ES.dat と STAT+SYS 共分散 | github.com/CobayaSampler/sn_data @ 61d9643 |
| data/sn/DESY5/ | DES-SN5YR_HD.csv と系統共分散 | 同上 |
| data/sn/DES-Dovekie/ | DES-Dovekie_HD.csv と逆共分散 | 同上 |
| data/sn/Union3/ | 22ビン圧縮データと共分散 | 同上 |

CMB 圧縮尤度の数値は src/likelihood.py の CMBCompressed に直接書いてある。github.com/cmbant/PlanckEarlyLCDM のチェーン（Lemos & Lewis, arXiv:2302.12911）から getdist で計算した値（ignore_rows=0.3）。

出典論文：DESI DR2 Results II（arXiv:2503.14738）、Pantheon+（Brout et al. 2022）、DES-SN5YR（DES Collaboration 2024）、Union3（Rubin et al., arXiv:2311.12098）、DES-Dovekie（Popovic et al.）。データを使う場合は各論文を引用すること。
