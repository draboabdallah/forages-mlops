"""Bonus - Tableau de bord des forages à risque (page HTML autonome, sans dépendance externe).

Usage : python -m forages.dashboard [--current data/forages_nouveau_mois.csv]
Produit reports/dashboard_forages.html : forages classés par probabilité de panne décroissante, filtres par région,
type de pompe et niveau de risque, et une capacité de visites réglable pour prioriser les tournées.
"""

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from string import Template

from forages import data
from forages.config import FEATURES, MODEL_PATH, NEW_PATH, REPORTS_DIR
from forages.predict import load_model

PAGE = Template("""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forages à risque</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --ink:#1d2433; --muted:#5b6475; --line:#e3e6ec;
        --high:#c0392b; --mid:#d68910; --low:#2e7d32; --accent:#2f5d8a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --card:#1d2129; --ink:#e7eaf0; --muted:#9aa3b2; --line:#2d333d;
          --high:#ef6f61; --mid:#f0b54a; --low:#6cc070; --accent:#7fb0e0; }
}
* { box-sizing:border-box; }
body { margin:0; font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
main { max-width:1150px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; }
.sub { color:var(--muted); margin:0 0 20px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-bottom:18px; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.kpi b { display:block; font-size:24px; font-variant-numeric:tabular-nums; }
.kpi span { color:var(--muted); font-size:12.5px; }
.panel { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
.controls { display:flex; flex-wrap:wrap; gap:14px; align-items:end; }
label { display:flex; flex-direction:column; gap:4px; font-size:12.5px; color:var(--muted); }
select, input { font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); }
.bars { display:grid; grid-template-columns:90px 1fr 70px; gap:6px 10px; align-items:center; }
.bar { height:12px; border-radius:6px; background:var(--accent); }
.wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
th { font-size:12px; color:var(--muted); font-weight:600; position:sticky; top:0; background:var(--card); }
tr.out td { opacity:.45; }
.pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; font-weight:600; color:#fff; }
.eleve { background:var(--high); } .moyen { background:var(--mid); } .faible { background:var(--low); }
.p { display:flex; align-items:center; gap:8px; }
.p i { display:inline-block; height:8px; border-radius:4px; background:var(--accent); }
footer { color:var(--muted); font-size:12px; margin-top:16px; }
</style>
</head>
<body>
<main>
  <h1>Forages à risque de panne dans les 90 jours</h1>
  <p class="sub">Lot <code>$source</code> · modèle <code>$version</code> · seuil de visite $threshold · généré le $generated</p>

  <div class="kpis">
    <div class="kpi"><b id="k-n"></b><span>forages affichés</span></div>
    <div class="kpi"><b id="k-visit"></b><span>au-dessus du seuil de visite</span></div>
    <div class="kpi"><b id="k-high"></b><span>risque élevé (≥ 60 %)</span></div>
    <div class="kpi"><b id="k-cap"></b><span>pannes attendues couvertes par les visites planifiées</span></div>
  </div>

  <div class="panel controls">
    <label>Région <select id="f-region"><option value="">Toutes</option></select></label>
    <label>Type de pompe <select id="f-pompe"><option value="">Tous</option></select></label>
    <label>Niveau de risque <select id="f-niveau"><option value="">Tous</option>
      <option value="eleve">Élevé</option><option value="moyen">Moyen</option><option value="faible">Faible</option></select></label>
    <label>Visites possibles <input id="f-cap" type="number" min="0" step="10" value="$capacity"></label>
  </div>

  <div class="panel">
    <div style="font-weight:600;margin-bottom:8px">Forages à visiter (au-dessus du seuil) par région</div>
    <div class="bars" id="bars"></div>
  </div>

  <div class="panel wrap">
    <table>
      <thead><tr><th>Rang</th><th>Forage</th><th>Probabilité</th><th>Niveau</th><th>Région</th><th>Pompe</th>
        <th>Âge</th><th>Usagers/j</th><th>Jours depuis maint.</th><th>Pannes 12 m</th><th>Débit m³/h</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <footer>Les lignes estompées dépassent la capacité de visites choisie. « Pannes attendues couvertes » = somme des
    probabilités des forages planifiés ÷ somme des probabilités de tous les forages affichés.</footer>
</main>
<script>
const DATA = $data;
const THRESHOLD = $threshold;
const $$ = (id) => document.getElementById(id);
const fmtPct = (x) => (100 * x).toFixed(0) + " %";
for (const [id, key] of [["f-region", "region"], ["f-pompe", "type_pompe"]]) {
  [...new Set(DATA.map(d => d[key]))].sort().forEach(v => $$(id).add(new Option(v, v)));
}
function render() {
  const region = $$("f-region").value, pompe = $$("f-pompe").value, niveau = $$("f-niveau").value;
  const cap = Math.max(0, parseInt($$("f-cap").value || "0", 10));
  const rows = DATA.filter(d => (!region || d.region === region) && (!pompe || d.type_pompe === pompe)
                              && (!niveau || d.niveau === niveau));
  const total = rows.reduce((s, d) => s + d.proba, 0);
  const planned = rows.slice(0, cap).reduce((s, d) => s + d.proba, 0);
  $$("k-n").textContent = rows.length;
  $$("k-visit").textContent = rows.filter(d => d.proba >= THRESHOLD).length;
  $$("k-high").textContent = rows.filter(d => d.niveau === "eleve").length;
  $$("k-cap").textContent = total ? fmtPct(planned / total) : "–";
  const byRegion = {};
  rows.filter(d => d.proba >= THRESHOLD).forEach(d => byRegion[d.region] = (byRegion[d.region] || 0) + 1);
  const max = Math.max(1, ...Object.values(byRegion));
  $$("bars").innerHTML = Object.entries(byRegion).sort((a, b) => b[1] - a[1]).map(([r, n]) =>
    `<span>$${r}</span><div class="bar" style="width:$${100 * n / max}%"></div><span>$${n}</span>`).join("");
  $$("rows").innerHTML = rows.map((d, i) => `<tr class="$${i < cap ? "" : "out"}"><td>$${i + 1}</td><td>$${d.id}</td>
    <td><span class="p"><i style="width:$${60 * d.proba}px"></i>$${fmtPct(d.proba)}</span></td>
    <td><span class="pill $${d.niveau}">$${d.niveau}</span></td><td>$${d.region}</td><td>$${d.type_pompe}</td>
    <td>$${d.age_ans}</td><td>$${d.usagers_jour}</td><td>$${d.jours_depuis_maintenance}</td><td>$${d.pannes_12m}</td>
    <td>$${d.debit_m3h}</td></tr>`).join("");
}
["f-region", "f-pompe", "f-niveau", "f-cap"].forEach(id => $$(id).addEventListener("input", render));
render();
</script>
</body>
</html>
""")


def build(current_path: Path = NEW_PATH, capacity: int = 100) -> Path:
    model = load_model(MODEL_PATH)
    df = data.load(current_path, require_target=False)
    df["proba"] = model.predict_proba(df).round(4)
    df["niveau"] = df["proba"].map(model.risk_level)
    df["id"] = [f"F-{i + 1:04d}" for i in range(len(df))]
    df = df.sort_values("proba", ascending=False)
    records = df[["id", "proba", "niveau"] + FEATURES].to_dict(orient="records")
    page = PAGE.substitute(
        source=html.escape(Path(current_path).name),
        version=html.escape(model.version),
        threshold=f"{model.threshold:.3f}",
        generated=datetime.now().strftime("%d/%m/%Y %H:%M"),
        capacity=capacity,
        data=json.dumps(records, ensure_ascii=False).replace("</", "<\\/"),
    )
    out = REPORTS_DIR / "dashboard_forages.html"
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", type=Path, default=NEW_PATH)
    parser.add_argument("--capacite", type=int, default=100, help="nombre de visites possibles (modifiable dans la page)")
    args = parser.parse_args()
    print(f"Tableau de bord : {build(args.current, args.capacite)}")


if __name__ == "__main__":
    main()
