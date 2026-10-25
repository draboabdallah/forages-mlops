"""Génère le support de soutenance (10 diapositives) à partir des rapports produits par les scripts.

Usage : python soutenance/make_slides.py   (après explore, train, drift et retrain)
Produit soutenance/soutenance.pptx. Aucun chiffre n'est saisi à la main : tout est lu dans reports/ et models/.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forages import data  # noqa: E402
from forages.config import TARGET  # noqa: E402
from forages.api.schemas import EXAMPLE  # noqa: E402
from forages.config import MODEL_PATH  # noqa: E402
from forages.explore import univariate_auc  # noqa: E402
from forages.predict import load_model  # noqa: E402

REPORTS, FIG = ROOT / "reports", ROOT / "reports" / "figures"
NAVY, ACCENT, RED, GREY, LIGHT = (RGBColor(0x1D, 0x24, 0x33), RGBColor(0x2F, 0x5D, 0x8A), RGBColor(0xC0, 0x39, 0x2B),
                                  RGBColor(0x5B, 0x64, 0x75), RGBColor(0xEE, 0xF2, 0xF7))
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
W, H = Inches(13.333), Inches(7.5)


def pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.n = 0

    def slide(self, title: str, kicker: str = ""):
        self.n += 1
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.12))
        band.fill.solid(), setattr(band.fill.fore_color, "rgb", ACCENT), band.line.fill.background()
        if kicker:
            self.text(s, kicker.upper(), 0.6, 0.35, 10, 0.3, size=11, color=ACCENT, bold=True)
        self.text(s, title, 0.6, 0.6, 12, 0.7, size=28, color=NAVY, bold=True)
        self.text(s, str(self.n), 12.4, 7.0, 0.5, 0.3, size=10, color=GREY, align=PP_ALIGN.RIGHT)
        return s

    @staticmethod
    def text(s, txt, x, y, w, h, size=16, color=NAVY, bold=False, align=PP_ALIGN.LEFT, font="Calibri"):
        box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        for i, line in enumerate(txt.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run()
            r.text = line
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
            r.font.color.rgb = color
        return box

    @staticmethod
    def bullets(s, items, x, y, w, h, size=16):
        box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            sub = item.startswith("  ")
            p.space_after = Pt(4 if sub else 8)
            p.level = 1 if sub else 0
            head, _, tail = item.strip().partition(" : ")
            if tail:
                r = p.add_run()
                r.text = ("– " if sub else "• ") + head + " : "
                r.font.bold = True
                r.font.size, r.font.color.rgb, r.font.name = Pt(size - (2 if sub else 0)), NAVY, "Calibri"
            r = p.add_run()
            r.text = tail if tail else ("– " if sub else "• ") + head
            r.font.size, r.font.color.rgb, r.font.name = Pt(size - (2 if sub else 0)), NAVY if not sub else GREY, "Calibri"
        return box

    @staticmethod
    def card(s, value, label, x, y, w=2.8, h=1.25, color=ACCENT):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        box.adjustments[0] = 0.08
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT
        box.line.fill.background()
        Deck.text(s, value, x + 0.15, y + 0.08, w - 0.3, 0.6, size=28, color=color, bold=True)
        Deck.text(s, label, x + 0.15, y + 0.68, w - 0.3, 0.55, size=12, color=GREY)

    @staticmethod
    def node(s, label, x, y, w=2.3, h=0.95, fill=ACCENT):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        box.fill.solid()
        box.fill.fore_color.rgb = fill
        box.line.fill.background()
        tf = box.text_frame
        tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.MIDDLE
        for i, line in enumerate(label.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = line
            r.font.size, r.font.bold, r.font.name = Pt(14 if i == 0 else 11), i == 0, "Calibri"
            r.font.color.rgb = WHITE
        return box

    @staticmethod
    def arrow(s, a, b, start=3, end=1):
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 0, 0, 0, 0)
        c.begin_connect(a, start), c.end_connect(b, end)
        c.line.color.rgb, c.line.width = GREY, Pt(2)
        c.line._get_or_add_ln().append(_arrow_head())

    @staticmethod
    def table(s, df: pd.DataFrame, x, y, w, row_h=0.42, size=13, highlight: int | None = None, widths=None):
        shape = s.shapes.add_table(len(df) + 1, len(df.columns), Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * (len(df) + 1)))
        t = shape.table
        if widths:
            for j, wj in enumerate(widths):
                t.columns[j].width = Inches(wj)
        for j, col in enumerate(df.columns):
            t.cell(0, j).text = str(col)
        for i, row in enumerate(df.itertuples(index=False), start=1):
            for j, v in enumerate(row):
                t.cell(i, j).text = str(v)
        for i in range(len(df) + 1):
            for j in range(len(df.columns)):
                cell = t.cell(i, j)
                for p in cell.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size, r.font.name = Pt(size), "Calibri"
                        r.font.bold = i == 0 or i == highlight
                        r.font.color.rgb = WHITE if i == 0 else NAVY
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xDC, 0xE8, 0xF5) if i == highlight else
                                                               (LIGHT if i % 2 else WHITE))
        return shape

    def image(self, s, path, x, y, w=None, h=None):
        kw = {}
        if w:
            kw["width"] = Inches(w)
        if h:
            kw["height"] = Inches(h)
        return s.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def _arrow_head():
    from pptx.oxml.ns import qn
    from lxml import etree

    el = etree.SubElement(etree.Element("dummy"), qn("a:tailEnd"))
    el.set("type", "triangle")
    return el


def count_tests() -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if "::" in line)


def main() -> None:
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    base = json.loads((REPORTS / "baseline.json").read_text(encoding="utf-8"))
    drift = json.loads((REPORTS / "drift_report.json").read_text(encoding="utf-8"))
    retrain = json.loads((REPORTS / "retrain_report.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(REPORTS / "model_runs.csv")
    hist = data.load()
    new = data.load(ROOT / "data" / "forages_nouveau_mois.csv")
    aucs = univariate_auc(hist)
    t, cv = meta["test_metrics"], meta["cv_metrics"]
    n_tests = count_tests()
    example_proba = float(load_model(MODEL_PATH).predict_proba(pd.DataFrame([EXAMPLE]))[0])
    d = Deck()

    # 1. Titre
    s = d.prs.slides.add_slide(d.prs.slide_layouts[6])
    d.n += 1
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid(), setattr(bg.fill.fore_color, "rgb", NAVY), bg.line.fill.background()
    d.text(s, "PROJET DE FIN DE COURS · APPRENTISSAGE AUTOMATIQUE APPLIQUÉ À LA PRODUCTION", 0.8, 1.6, 11.5, 0.4,
           size=13, color=RGBColor(0x9F, 0xC3, 0xE6), bold=True)
    d.text(s, "Du modèle à la production :\nprédire les pannes de forages", 0.8, 2.1, 11.5, 1.9, size=44,
           color=WHITE, bold=True)
    d.text(s, "Un service de prédiction tracé, testé, conteneurisé et surveillé, pour planifier les visites "
              "préventives des équipes de maintenance.", 0.8, 4.2, 10.5, 0.9, size=18, color=RGBColor(0xD5, 0xDD, 0xE8))
    d.text(s, "Groupe : Muhammed Ramadane KOMI · Amir DRABO · Mohamed KABORE\nEnseignant : Dr TIEMTORE Souleymane · octobre 2026", 0.8, 5.9, 11.5, 0.9, size=14,
           color=RGBColor(0x9F, 0xC3, 0xE6))

    # 2. Problème et métrique
    s = d.slide("Le besoin : savoir quels forages visiter avant la panne", "Problème et métrique")
    d.bullets(s, [
        f"Données : {len(hist):,} forages historiques".replace(",", " ") + f", {len(new)} forages d'un mois récent",
        "  8 variables : âge, profondeur, débit, usagers, maintenance, pannes passées, pompe, région",
        f"Cible : panne dans les 90 jours ({pct(hist[TARGET].mean())} des forages)",
        f"Piège : un modèle qui ne prédit jamais de panne a {pct(1 - hist[TARGET].mean())} d'accuracy",
        "Coût asymétrique : rater une panne = des semaines sans eau ; visite inutile = une journée d'équipe",
    ], 0.6, 1.6, 6.6, 5)
    d.card(s, "PR-AUC", "métrique de sélection : qualité du classement des forages par risque", 7.7, 1.7, 4.9)
    d.card(s, f"rappel ≥ {pct(meta['target_recall'])}", "le seuil de décision est fixé pour ne pas rater plus "
           "d'une panne sur cinq", 7.7, 3.15, 4.9, color=RED)
    d.card(s, "précision, F2, Brier", "suivis en complément : charge de visites inutiles et calibration",
           7.7, 4.6, 4.9, color=GREY)

    # 3. Exploration et fuites
    s = d.slide("Exploration : des signaux modestes, pas de fuite", "J1 · Données")
    d.image(s, FIG / "distributions.png", 0.5, 1.5, w=7.4)
    top = aucs.head(4)
    d.bullets(s, [
        "Variables les plus liées à la panne (AUC seule) :",
        *[f"  {k} : {num(v, 2)}" for k, v in top.items()],
        "Aucune variable ne sépare presque parfaitement les classes : pas de fuite évidente",
        "Prétraitement dans le pipeline, test isolé de la sélection",
        "À valider avec le métier : pannes_12m et maintenance mesurées avant la fenêtre de 90 jours",
    ], 8.2, 1.5, 4.7, 5.5, size=14)

    # 4. Essais MLflow
    s = d.slide(f"{len(runs)} essais suivis dans MLflow, le plus simple l'emporte", "J2 · Modèle")
    order = runs.sort_values("cv_pr_auc", ascending=False)
    tab = pd.DataFrame({
        "Modèle": order["modele"],
        "PR-AUC CV": [f"{num(a)} ± {num(b)}" for a, b in zip(order["cv_pr_auc"], order["cv_pr_auc_std"])],
        "PR-AUC test": order["test_pr_auc"].map(num),
        "Rappel test": order["test_rappel"].map(num),
        "Brier test": order["test_brier"].map(num),
    })
    hl = list(order["modele"]).index(meta["model_name"]) + 1
    d.table(s, tab, 0.6, 1.55, 7.6, highlight=hl, widths=[2.6, 1.6, 1.2, 1.1, 1.1])
    d.bullets(s, [
        f"Retenu : {meta['model_name']} (régression logistique)",
        "Règle : meilleure PR-AUC en validation croisée ; à 0,005 près, le plus simple gagne",
        "Les arbres et les splines n'apportent rien : relation quasi linéaire",
        f"Logistique : lisible et la mieux calibrée (Brier {num(t['brier'])})",
        f"Registre MLflow : {meta['registry_name']} v{meta['registry_version']}, alias « champion »",
    ], 8.6, 1.5, 4.4, 5.5, size=14)

    # 5. Modèle retenu
    s = d.slide(f"Sur 100 pannes à venir, le modèle en signale {round(t['recall'] * 100)}",
                "J2 · Performance sur le test")
    d.card(s, num(t["pr_auc"]), f"PR-AUC test (hasard : {num(base['dummy_prior']['pr_auc'])})", 0.6, 1.6)
    d.card(s, pct(t["recall"]), "rappel : pannes détectées", 0.6, 3.0, color=RED)
    d.card(s, pct(t["precision"]), "précision : visites utiles", 0.6, 4.4, color=GREY)
    d.text(s, f"Seuil de décision {num(meta['threshold'])}, choisi en validation croisée et appliqué tel quel "
              "au jeu de test.", 0.6, 5.85, 2.9, 1.2, size=11, color=GREY)
    d.image(s, FIG / "courbes_pr.png", 3.7, 1.5, h=4.6)
    d.image(s, FIG / "importance_variables.png", 9.15, 2.0, w=3.9)

    # 6. Architecture
    s = d.slide("Architecture : du CSV au service surveillé", "Industrialisation")
    n_data = d.node(s, "Données\nCSV validés (schéma)", 0.5, 2.0, fill=GREY)
    n_train = d.node(s, "Entraînement\nforages.train · CV 5 plis", 3.3, 2.0)
    n_mlflow = d.node(s, "MLflow\nruns · champion / challenger", 3.3, 3.75, fill=NAVY)
    n_model = d.node(s, "Modèle livré\nmodel.joblib + metadata", 6.1, 2.0)
    n_api = d.node(s, "API FastAPI\n/v1/predict · /health", 8.9, 2.0)
    n_docker = d.node(s, "Image Docker\npython 3.12, non root", 8.9, 3.75, fill=NAVY)
    n_drift = d.node(s, "Surveillance\nforages.drift · retrain", 6.1, 3.75, fill=RED)
    for a, b in [(n_data, n_train), (n_train, n_model), (n_model, n_api)]:
        d.arrow(s, a, b)
    # points de connexion : 0 haut, 1 gauche, 2 bas, 3 droite
    d.arrow(s, n_train, n_mlflow, 2, 0)
    d.arrow(s, n_api, n_docker, 2, 0)
    d.arrow(s, n_model, n_drift, 2, 0)
    d.arrow(s, n_drift, n_mlflow, 1, 3)
    ci = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(5.3), Inches(10.7), Inches(0.75))
    ci.fill.solid(), setattr(ci.fill.fore_color, "rgb", LIGHT), ci.line.fill.background()
    d.text(s, "CI GitHub Actions à chaque push : tests → ré-entraînement et contrôle de reproductibilité → garde-fou "
              "de performance → rapport de dérive → construction et test du conteneur", 0.7, 5.4, 10.3, 0.6,
           size=13, color=NAVY)
    d.text(s, "Traçabilité : chaque réponse de l'API porte la version du modèle, qui renvoie au run MLflow, "
              "au commit Git et à l'empreinte des données.", 0.5, 6.25, 11.5, 0.6, size=13, color=GREY)

    # 7. API
    s = d.slide("API : un contrat strict et versionné", "J3 · Service")
    d.table(s, pd.DataFrame({
        "Route": ["GET /health", "GET /v1/model", "POST /v1/predict", "POST /predict"],
        "Rôle": ["état et version du modèle chargé", "run MLflow, seuil, métriques",
                 "lot de 1 à 1 000 forages", "alias de /v1/predict"],
    }), 0.6, 1.55, 6.2, row_h=0.45, size=13)
    d.bullets(s, [
        "Validation Pydantic : types, bornes physiques, modalités, champs inconnus refusés → 422",
        "Versionnement : /v1 dans l'URL, version du modèle dans la réponse et l'en-tête X-Model-Version",
        "Sans modèle : mode dégradé signalé par /health, 503 sur les prédictions",
    ], 0.6, 4.1, 6.2, 3, size=14)
    code = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.2), Inches(1.55), Inches(5.6), Inches(5.1))
    code.fill.solid(), setattr(code.fill.fore_color, "rgb", NAVY), code.line.fill.background()
    d.text(s, 'POST /v1/predict\n{"forages": [{\n  "forage_id": "F-0001",\n  "age_ans": 22,\n'
              '  "jours_depuis_maintenance": 420,\n  "pannes_12m": 2,\n  "type_pompe": "thermique", ...}]}\n\n'
              f'→ 200\n{{"model_version": "{meta["model_version"]}",\n "seuil": {meta["threshold"]:.4f},\n'
              ' "predictions": [{"forage_id": "F-0001",\n'
              f'   "probabilite_panne": {example_proba:.4f},\n'
              '   "a_visiter": true, "niveau_risque": "eleve"}]}',
           7.4, 1.7, 5.3, 4.9, size=12, color=RGBColor(0xD5, 0xDD, 0xE8), font="Consolas")

    # 8. Qualité
    s = d.slide("Qualité : tests, intégration continue et garde-fou", "J3-J4 · Tests et CI")
    d.card(s, str(n_tests), "tests pytest (données, métriques, pipeline, API, dérive, performance)", 0.6, 1.6, 3.6)
    d.card(s, "Reproductible", "la CI ré-entraîne depuis zéro et compare aux métriques livrées", 0.6, 3.05, 3.6)
    d.card(s, "Docker", "image construite, démarrée et interrogée (CI et en local)", 0.6, 4.5, 3.6, color=NAVY)
    d.bullets(s, [
        "Garde-fou de performance : la CI échoue si le modèle passe sous",
        "  PR-AUC 0,55 · ROC-AUC 0,70 · rappel 0,75 · gain sur le hasard 0,15",
        "Seuils = engagements métier, pas les scores actuels : marge pour la variance, blocage des régressions",
        "Tests de l'API : cas valides, lots, ordre, alias, 8 entrées invalides, mode dégradé",
        "Tests des données : schéma, bornes, modalités, découpage stratifié et reproductible",
    ], 4.7, 1.6, 8.2, 5.2, size=15)

    # 9. Surveillance
    s = d.slide(f"Nouveau mois : dérive détectée, décision {drift['decision']}", "J5 · Surveillance")
    d.image(s, FIG / "derive_distributions.png", 0.4, 1.4, w=12.5)
    var0 = drift["derive_variables"][0]
    perf = drift["performance_courante"]
    d.bullets(s, [
        f"Dérive : {var0['variable']} de {var0['ref_moyenne']:.0f} à {var0['courant_moyenne']:.0f} "
        f"(PSI {num(var0['psi'], 2)}), {pct(var0['hors_plage'])} hors plage d'entraînement",
        f"Mais le modèle tient : ROC-AUC {num(perf['roc_auc'])}, rappel {num(perf['recall'])}, "
        "écart de calibration " + f"{drift['ecart_calibration']:+.3f}".replace(".", ",").replace("-", "−"),
    ], 0.6, 4.75, 6.3, 2.3, size=14)
    d.bullets(s, [
        f"Champion / challenger : gain de PR-AUC {retrain['gain_pr_auc']:+.3f}, IC 95 % "
        f"[{retrain['gain_ic95'][0]:+.3f} ; {retrain['gain_ic95'][1]:+.3f}]".replace(".", ","),
        f"Seuil de promotion {retrain['min_gain']:+.2f} ".replace(".", ",")
        + ("atteint : " if retrain["promu"] else "non atteint : ")
        + ("challenger promu" if retrain["promu"] else "champion conservé"),
        f"Impact : {pct(drift['derive_predictions']['courant_taux_alerte'])} des forages au-dessus du seuil "
        "→ prioriser par probabilité",
    ], 7.0, 4.75, 6.0, 2.3, size=14)

    # 10. Bonus et suite
    s = d.slide("Bonus, limites et prochaines étapes", "Conclusion")
    d.bullets(s, [
        "Tableau de bord : forages classés par risque, filtres, capacité de visites réglable",
        "Registre MLflow : alias champion / challenger, promotion conditionnée à un gain mesuré",
        "Règle de décision : OK / SURVEILLER / RE-ENTRAINER, testée et exécutée par la CI",
    ], 0.6, 1.6, 6.0, 4, size=15)
    d.bullets(s, [
        "Limites : données fictives, un seul mois récent étiqueté, hypothèse temporelle à confirmer",
        "Capacité : au seuil actuel, trop de forages signalés pour toutes les équipes",
        "Suite : dérive mensuelle planifiée (--fail-on-retrain), second mois pour le challenger, "
        "seuil fixé selon la capacité réelle de visites",
    ], 7.0, 1.6, 5.9, 4, size=15)
    d.text(s, "Tout résultat présenté est produit par un script du dépôt : explore → train → drift → retrain → "
              "dashboard → make_slides.", 0.6, 6.3, 12, 0.5, size=13, color=GREY)

    out = ROOT / "soutenance" / "soutenance.pptx"
    d.prs.save(out)
    print(f"{d.n} diapositives -> {out}")


if __name__ == "__main__":
    main()
