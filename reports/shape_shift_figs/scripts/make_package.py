"""Assemble the delivery folder and the source archive after build_figures.py.

    python scripts/make_package.py      # from the package root

Delivery (../shape_shift_figs_delivery/):
  PDF_矢量/            one PDF per figure        预览/   PNG previews
  形状偏移图表合集.pdf  every page in one file    附表/   MPC steps table (PDF + PNG)
  Python绘图源码与数据.zip  scripts + data + frames: rebuild offline with  python scripts/build_figures.py
  说明.txt, English_captions.txt, 图表清单.json, 图表索引.html, 文件校验_SHA256.json
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEL = ROOT.parent / "shape_shift_figs_delivery"
if DEL.exists():
    shutil.rmtree(DEL)
(DEL / "PDF_矢量").mkdir(parents=True)
(DEL / "预览").mkdir()
(DEL / "附表").mkdir()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


manifest = json.loads((ROOT / "figure_manifest.json").read_text(encoding="utf-8"))
for x in manifest:
    if x["name"].endswith("mpc_steps_table"):
        shutil.copy2(ROOT / "figures" / f"{x['name']}.pdf", DEL / "附表" / "MPC平均步数对比表.pdf")
        shutil.copy2(ROOT / "preview" / f"{x['name']}.png", DEL / "附表" / "MPC平均步数对比表.png")
    shutil.copy2(ROOT / "figures" / f"{x['name']}.pdf", DEL / "PDF_矢量")
    shutil.copy2(ROOT / "preview" / f"{x['name']}.png", DEL / "预览")
for name in ["形状偏移图表合集.pdf", "说明.txt", "English_captions.txt"]:
    shutil.copy2(ROOT / name, DEL / name)
shutil.copy2(ROOT / "figure_manifest.json", DEL / "图表清单.json")
page = (ROOT / "图表索引.html").read_text(encoding="utf-8")
page = page.replace('href="figures/', 'href="PDF_矢量/').replace('src="preview/', 'src="预览/')
(DEL / "图表索引.html").write_text(page, encoding="utf-8")

src_items = ["scripts", "data", "replay", "thumbnails", "figures", "preview", "figure_manifest.json",
             "English_captions.txt", "说明.txt", "图表索引.html", "形状偏移图表合集.pdf"]
zpath = DEL / "Python绘图源码与数据.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for item in src_items:
        p = ROOT / item
        files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file() and "__pycache__" not in q.parts)
        for q in files:
            z.write(q, Path("shape_shift_figs") / q.relative_to(ROOT))
sums = {str(p.relative_to(DEL)): sha(p) for p in sorted(DEL.rglob("*")) if p.is_file()}
(DEL / "文件校验_SHA256.json").write_text(json.dumps(sums, indent=2, ensure_ascii=False), encoding="utf-8")
print("delivery:", DEL, len(sums), "files; zip", round(zpath.stat().st_size / 2 ** 20, 1), "MB")
