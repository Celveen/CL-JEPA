"""Fill template.html with report_data.json and narrative.json -> report.html (all in reports/cmp)."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
t = (H / "template.html").read_text(encoding="utf-8")
n = json.loads((H / "narrative.json").read_text(encoding="utf-8"))
data = (H / "report_data.json").read_text(encoding="utf-8").replace("</", "<\\/")
t = (t.replace("__LEAD__", n["lead"])
      .replace("__FINDINGS__", "".join(f"<li>{x}</li>" for x in n["findings"]))
      .replace("__SETUP__", "".join(f"<p>{x}</p>" for x in n["setup"]))
      .replace("__LIMITS__", "".join(f"<li>{x}</li>" for x in n["limits"]))
      .replace("__DATA__", data))
assert "__" not in t.split('<script id="data"')[0], "unfilled placeholder"
(H / "report.html").write_text(t, encoding="utf-8")
print(H / "report.html", f"{len(t.encode()) / 1024:.0f} KB")
