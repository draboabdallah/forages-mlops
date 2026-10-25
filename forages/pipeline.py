"""Construction des pipelines scikit-learn (prétraitement + modèle dans un seul objet).

Tout le prétraitement est appris à l'intérieur du pipeline : il est ré-ajusté à chaque pli de
validation croisée et n'est jamais ajusté sur le jeu de test, ce qui évite la fuite de données.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from forages.config import CAT_FEATURES, CATEGORIES, NUM_FEATURES


def numeric_transformer(kind: str):
    if kind == "scale":
        return StandardScaler()
    if kind == "splines":
        # Effets non linéaires par variable (ex. risque qui sature avec l'âge), restant additifs.
        return Pipeline([("splines", SplineTransformer(n_knots=5, degree=3)), ("scale", StandardScaler())])
    if kind == "passthrough":
        return "passthrough"
    raise ValueError(f"Transformation numérique inconnue : {kind}")


def build_preprocessor(numeric: str = "scale") -> ColumnTransformer:
    cat = OneHotEncoder(
        categories=[CATEGORIES[c] for c in CAT_FEATURES],
        handle_unknown="ignore",
        sparse_output=False,
    )
    return ColumnTransformer(
        [("num", numeric_transformer(numeric), NUM_FEATURES), ("cat", cat, CAT_FEATURES)],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, numeric: str = "scale") -> Pipeline:
    return Pipeline([("prep", build_preprocessor(numeric)), ("model", estimator)])
