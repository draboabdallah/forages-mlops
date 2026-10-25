"""Petits utilitaires de génération de rapports Markdown (sans dépendance supplémentaire)."""

import pandas as pd


def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.3f}") -> str:
    if index:
        df = df.rename_axis(df.index.name or "variable").reset_index()
    cells = [[floatfmt.format(v) if isinstance(v, float) else str(v) for v in row] for row in df.itertuples(index=False)]
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in cells)
    return "\n".join([header, sep, body])
