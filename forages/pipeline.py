"""Construction des pipelines scikit-learn (prétraitement + modèle dans un seul objet).

Tout le prétraitement est appris à l'intérieur du pipeline : il est ré-ajusté à chaque pli de
validation croisée et n'est jamais ajusté sur le jeu de test, ce qui évite la fuite de données.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forages.config import CAT_FEATURES, CATEGORIES, NUM_FEATURES


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    cat = OneHotEncoder(
        categories=[CATEGORIES[c] for c in CAT_FEATURES],
        handle_unknown="ignore",
        sparse_output=False,
    )
    return ColumnTransformer(
        [("num", num, NUM_FEATURES), ("cat", cat, CAT_FEATURES)],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, scale: bool = True) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(scale)), ("model", estimator)])
