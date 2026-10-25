"""J5 - Surveillance : dérive des données, des prédictions et des performances, et règle de ré-entraînement.

Usage : python -m forages.drift [--current data/forages_nouveau_mois.csv] [--fail-on-retrain]
Produit reports/drift_report.md, reports/drift_report.json et reports/figures/derive_*.png.

Référence = données d'entraînement du modèle (train de l'historique). Courant = nouveau lot de forages.

Règle de décision (évaluée dans cet ordre) :
- RE-ENTRAINER si au moins une des conditions suivantes est vraie :
    * une variable a un PSI >= 0,25 (dérive majeure de la population) ;
    * les étiquettes sont connues et le rappel au seuil < 0,75 (on rate plus d'une panne sur quatre) ;
    * les étiquettes sont connues et la ROC-AUC a perdu plus de 0,05 par rapport au test ;
    * le taux de panne observé s'écarte de plus de 0,05 de la probabilité moyenne prédite (modèle mal calibré).
- SURVEILLER si une variable ou les prédictions ont un PSI entre 0,10 et 0,25.
- OK sinon.
Les étiquettes arrivent 90 jours après la prédiction : la dérive des données et des prédictions est le signal
précoce, la performance est la confirmation. La ROC-AUC sert de comparaison car, contrairement à la PR-AUC, elle ne
dépend pas du taux de panne, qui peut lui-même varier d'un mois à l'autre.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from forages import data
from forages.config import CAT_FEATURES, FIGURES_DIR, MODEL_PATH, NEW_PATH, NUM_FEATURES, REPORTS_DIR, TARGET
from forages.metrics import compute_metrics
from forages.predict import load_model
from forages.reporting import md_table

PSI_WARN = 0.10
PSI_ALERT = 0.25
MIN_RECALL = 0.75
MAX_ROC_AUC_DROP = 0.05
MAX_CALIBRATION_GAP = 0.05
EPS = 1e-4


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index sur les déciles de la référence."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    return _psi(ref_pct, cur_pct)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref) | set(cur))
    ref_pct = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur_pct = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref_pct, cur_pct)


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    ref_pct = np.clip(ref_pct, EPS, None)
    cur_pct = np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_level(psi: float) -> str:
    return "majeure" if psi >= PSI_ALERT else "modérée" if psi >= PSI_WARN else "stable"


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUM_FEATURES:
        rows.append({"variable": col, "type": "numérique", "ref_moyenne": float(ref[col].mean()),
                     "courant_moyenne": float(cur[col].mean()), "psi": psi_numeric(ref[col], cur[col]),
                     "p_value": float(ks_2samp(ref[col], cur[col]).pvalue), "test": "KS"})
    for col in CAT_FEATURES:
        table = pd.concat([ref[col].value_counts(), cur[col].value_counts()], axis=1).fillna(0)
        top_ref = ref[col].value_counts(normalize=True)
        top_cur = cur[col].value_counts(normalize=True).reindex(top_ref.index, fill_value=0)
        rows.append({"variable": col, "type": "catégorielle",
                     "ref_moyenne": float(top_ref.iloc[0]), "courant_moyenne": float(top_cur.iloc[0]),
                     "psi": psi_categorical(ref[col], cur[col]),
                     "p_value": float(chi2_contingency(table.to_numpy())[1]), "test": "chi²"})
    df = pd.DataFrame(rows).set_index("variable")
    df["niveau"] = df["psi"].map(psi_level)
    return df.sort_values("psi", ascending=False)


def decide(features: pd.DataFrame, pred_psi: float, perf: dict | None, ref_perf: dict, calibration_gap: float | None):
    reasons_retrain, reasons_watch = [], []
    for var, row in features.iterrows():
        if row["psi"] >= PSI_ALERT:
            reasons_retrain.append(f"dérive majeure de `{var}` (PSI {row['psi']:.2f} ≥ {PSI_ALERT})")
        elif row["psi"] >= PSI_WARN:
            reasons_watch.append(f"dérive modérée de `{var}` (PSI {row['psi']:.2f})")
    if pred_psi >= PSI_ALERT:
        reasons_retrain.append(f"dérive majeure des prédictions (PSI {pred_psi:.2f})")
    elif pred_psi >= PSI_WARN:
        reasons_watch.append(f"dérive modérée des prédictions (PSI {pred_psi:.2f})")
    if perf is not None:
        if perf["recall"] < MIN_RECALL:
            reasons_retrain.append(f"rappel {perf['recall']:.2f} < {MIN_RECALL}")
        drop = ref_perf["roc_auc"] - perf["roc_auc"]
        if drop > MAX_ROC_AUC_DROP:
            reasons_retrain.append(f"ROC-AUC en baisse de {drop:.3f} (> {MAX_ROC_AUC_DROP})")
        if abs(calibration_gap) > MAX_CALIBRATION_GAP:
            reasons_retrain.append(f"écart de calibration {calibration_gap:+.3f} (taux observé - probabilité "
                                   f"moyenne prédite, seuil ±{MAX_CALIBRATION_GAP})")
    decision = "RE-ENTRAINER" if reasons_retrain else "SURVEILLER" if reasons_watch else "OK"
    return decision, reasons_retrain, reasons_watch


def plot_drift(ref: pd.DataFrame, cur: pd.DataFrame, features: pd.DataFrame, ref_proba, cur_proba) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = [v for v in features.index if v in NUM_FEATURES][:3]
    fig, axes = plt.subplots(1, len(top) + 1, figsize=(4.2 * (len(top) + 1), 3.6))
    for ax, col in zip(axes, top):
        ax.hist(ref[col], bins=30, alpha=0.55, density=True, label="référence", color="#4C78A8")
        ax.hist(cur[col], bins=30, alpha=0.55, density=True, label="nouveau mois", color="#E45756")
        ax.set_title(f"{col} (PSI {features.loc[col, 'psi']:.2f})")
    axes[-1].hist(ref_proba, bins=30, alpha=0.55, density=True, label="référence", color="#4C78A8")
    axes[-1].hist(cur_proba, bins=30, alpha=0.55, density=True, label="nouveau mois", color="#E45756")
    axes[-1].set_title("probabilité prédite")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "derive_distributions.png", dpi=110)
    plt.close(fig)


def run(current_path: Path = NEW_PATH, model_path: Path = MODEL_PATH) -> dict:
    model = load_model(model_path)
    X_ref, _, _, _ = data.split(data.load())
    has_target = TARGET in pd.read_csv(current_path, nrows=0).columns
    cur = data.load(current_path, require_target=has_target)

    features = feature_drift(X_ref, cur)
    ref_proba, cur_proba = model.predict_proba(X_ref), model.predict_proba(cur)
    pred_psi = psi_numeric(pd.Series(ref_proba), pd.Series(cur_proba))
    ref_perf = model.metadata["test_metrics"]
    perf = compute_metrics(cur[TARGET], cur_proba, model.threshold) if has_target else None
    calibration_gap = float(cur[TARGET].mean() - cur_proba.mean()) if has_target else None
    decision, retrain, watch = decide(features, pred_psi, perf, ref_perf, calibration_gap)
    plot_drift(X_ref, cur, features, ref_proba, cur_proba)

    result = {
        "fichier_courant": Path(current_path).name,
        "model_version": model.version,
        "n_reference": len(X_ref),
        "n_courant": len(cur),
        "decision": decision,
        "raisons_reentrainement": retrain,
        "points_de_vigilance": watch,
        "derive_variables": features.reset_index().to_dict(orient="records"),
        "derive_predictions": {"psi": pred_psi, "ref_proba_moyenne": float(ref_proba.mean()),
                               "courant_proba_moyenne": float(cur_proba.mean()),
                               "ref_taux_alerte": float((ref_proba >= model.threshold).mean()),
                               "courant_taux_alerte": float((cur_proba >= model.threshold).mean())},
        "performance_reference_test": ref_perf,
        "performance_courante": perf,
        "taux_panne_observe": float(cur[TARGET].mean()) if has_target else None,
        "ecart_calibration": calibration_gap,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "drift_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS_DIR / "drift_report.md").write_text(render(result, features), encoding="utf-8")
    return result


def render(r: dict, features: pd.DataFrame) -> str:
    p = r["derive_predictions"]
    perf_md = "Étiquettes absentes : performance non mesurable sur ce lot."
    if r["performance_courante"]:
        keys = ["pr_auc", "roc_auc", "recall", "precision", "f2", "brier", "taux_alerte"]
        perf_df = pd.DataFrame({"test (référence)": {k: r["performance_reference_test"][k] for k in keys},
                                "nouveau mois": {k: r["performance_courante"][k] for k in keys}})
        perf_df.index.name = "métrique"
        perf_md = (md_table(perf_df) + f"\n\nTaux de panne observé : **{r['taux_panne_observe']:.3f}** ; "
                   f"probabilité moyenne prédite : **{p['courant_proba_moyenne']:.3f}** ; écart de calibration : "
                   f"**{r['ecart_calibration']:+.3f}**.\n\nLa PR-AUC dépend du taux de panne (plus élevé ce mois-ci) : "
                   f"la comparaison se fait sur la ROC-AUC, le rappel et la calibration.")
    bullets = lambda items: "\n".join(f"- {x}" for x in items) or "- aucun"
    table = features.drop(columns="type").rename(columns={"ref_moyenne": "réf. (moy. ou part)",
                                                          "courant_moyenne": "courant"})
    return f"""# J5 - Rapport de dérive

*Généré par `python -m forages.drift`. Ne pas modifier à la main.*

Modèle `{r['model_version']}` ; référence : {r['n_reference']} forages d'entraînement ; lot courant :
`{r['fichier_courant']}` ({r['n_courant']} forages).

## Décision : **{r['decision']}**

Raisons de ré-entraîner :
{bullets(r['raisons_reentrainement'])}

Points de vigilance :
{bullets(r['points_de_vigilance'])}

## Dérive des variables

PSI : < {PSI_WARN} stable, {PSI_WARN} à {PSI_ALERT} modérée, ≥ {PSI_ALERT} majeure. Test KS (numériques) ou du chi²
(catégorielles) : une p-value < 0,05 signale une différence de distribution statistiquement significative. Pour les
catégorielles, les colonnes de moyenne donnent la part de la modalité la plus fréquente.

{md_table(table)}

## Dérive des prédictions

| | référence | nouveau mois |
|---|---|---|
| probabilité moyenne | {p['ref_proba_moyenne']:.3f} | {p['courant_proba_moyenne']:.3f} |
| part de forages à visiter | {p['ref_taux_alerte']:.3f} | {p['courant_taux_alerte']:.3f} |

PSI des probabilités prédites : **{p['psi']:.3f}** ({psi_level(p['psi'])}).

![Dérive](figures/derive_distributions.png)

## Performance sur le lot courant

{perf_md}

## Règle de ré-entraînement

- **RE-ENTRAINER** si une variable a un PSI ≥ {PSI_ALERT}, ou, une fois les étiquettes connues (90 jours plus tard),
  si le rappel au seuil passe sous {MIN_RECALL}, si la ROC-AUC perd plus de {MAX_ROC_AUC_DROP} par rapport au test, ou
  si l'écart entre taux de panne observé et probabilité moyenne prédite dépasse ±{MAX_CALIBRATION_GAP}.
- **SURVEILLER** si une variable ou les prédictions ont un PSI entre {PSI_WARN} et {PSI_ALERT}.
- **OK** sinon.

Le ré-entraînement se fait avec `python -m forages.retrain` (historique + données récentes étiquetées). Le modèle
challenger n'est promu champion que s'il fait mieux que le champion actuel sur des données récentes qu'il n'a pas vues.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", type=Path, default=NEW_PATH, help="lot de forages à comparer à la référence")
    parser.add_argument("--fail-on-retrain", action="store_true", help="code de sortie 1 si la décision est RE-ENTRAINER")
    args = parser.parse_args()
    result = run(args.current)
    print(f"Décision : {result['decision']}")
    for reason in result["raisons_reentrainement"] + result["points_de_vigilance"]:
        print(f"  - {reason}")
    print(f"Rapport : {REPORTS_DIR / 'drift_report.md'}")
    if args.fail_on_retrain and result["decision"] == "RE-ENTRAINER":
        sys.exit(1)


if __name__ == "__main__":
    main()
