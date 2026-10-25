"""Chargement, validation et découpage des données."""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from forages.config import (
    BOUNDS,
    CATEGORIES,
    FEATURES,
    HIST_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def validate(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Vérifie le schéma attendu ; lève ValueError avec un message explicite sinon."""
    expected = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    nulls = df[expected].isna().sum()
    if nulls.any():
        raise ValueError(f"Valeurs manquantes : {nulls[nulls > 0].to_dict()}")
    for col, (low, high) in BOUNDS.items():
        out = df[(df[col] < low) | (df[col] > high)]
        if len(out):
            raise ValueError(f"{len(out)} valeur(s) hors bornes [{low}, {high}] pour {col}")
    for col, allowed in CATEGORIES.items():
        unknown = set(df[col].unique()) - set(allowed)
        if unknown:
            raise ValueError(f"Modalités inconnues pour {col} : {sorted(unknown)}")
    if require_target and not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"La cible {TARGET} doit être binaire 0/1")
    return df


def load(path: Path = HIST_PATH, require_target: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in CATEGORIES:
        df[col] = df[col].astype(str)
    return validate(df, require_target=require_target)


def split(df: pd.DataFrame):
    """Découpage stratifié train/test, figé par RANDOM_STATE. Le test n'est utilisé qu'en fin de parcours."""
    return train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )


def file_hash(path: Path) -> str:
    """Empreinte SHA-256 du fichier, tracée dans MLflow pour savoir sur quelles données un modèle a appris."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
