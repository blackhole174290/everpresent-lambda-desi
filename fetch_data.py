"""公開データを取得して data/ に置く（Windows・Mac・Linux 共通、git は不要）。
再現性のため、取得元リポジトリのコミットを固定している。
使い方: python fetch_data.py"""
import io, os, shutil, urllib.request, zipfile

SOURCES = [
    # (リポジトリ, コミット, リポジトリ内のフォルダ, 置き先)
    ("CobayaSampler/bao_data", "bb0c1c9009dc76d1391300e169e8df38fd1096db", "desi_bao_dr2", "data/desi_dr2"),
    ("CobayaSampler/sn_data", "61d96434cafc2770928322c38e5a750e686368ae", "PantheonPlus", "data/sn/PantheonPlus"),
    ("CobayaSampler/sn_data", "61d96434cafc2770928322c38e5a750e686368ae", "DESY5", "data/sn/DESY5"),
    ("CobayaSampler/sn_data", "61d96434cafc2770928322c38e5a750e686368ae", "Union3", "data/sn/Union3"),
    ("CobayaSampler/sn_data", "61d96434cafc2770928322c38e5a750e686368ae", "DES-Dovekie", "data/sn/DES-Dovekie"),
]
NEEDED_BAO = {"desi_gaussian_bao_ALL_GCcomb_mean.txt", "desi_gaussian_bao_ALL_GCcomb_cov.txt"}

_cache = {}
def _zip(repo, commit):
    if (repo, commit) not in _cache:
        url = f"https://codeload.github.com/{repo}/zip/{commit}"
        print(f"ダウンロード中: {repo} @ {commit[:7]}")
        with urllib.request.urlopen(url) as r:
            _cache[(repo, commit)] = zipfile.ZipFile(io.BytesIO(r.read()))
    return _cache[(repo, commit)]

for repo, commit, folder, dest in SOURCES:
    z = _zip(repo, commit)
    prefix = f"{repo.split('/')[1]}-{commit}/{folder}/"
    os.makedirs(dest, exist_ok=True)
    n = 0
    for name in z.namelist():
        if name.startswith(prefix) and not name.endswith("/"):
            base = os.path.basename(name)
            if folder == "desi_bao_dr2" and base not in NEEDED_BAO:
                continue
            with z.open(name) as src, open(os.path.join(dest, base), "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    print(f"  {folder} → {dest}（{n} ファイル）")
print("完了")
