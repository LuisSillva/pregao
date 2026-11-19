"""Acesso aos preços pelo app, com cache em memória.

Ler o parquet a cada clique seria desperdício, mas guardar para sempre faria o
servidor ignorar os dados novos que o robô publica todo dia. A saída: guardar
em memória e recarregar só quando o arquivo mudar (data de modificação diferente).
"""

from __future__ import annotations

import pandas as pd

from pregao import data

_cache: dict = {"mtime": None, "df": None}


def prices() -> pd.DataFrame:
    path = data.DATA_PATH
    mtime = path.stat().st_mtime if path.exists() else None
    if _cache["df"] is None or mtime != _cache["mtime"]:
        _cache["df"] = data.load(path)
        _cache["mtime"] = mtime
    return _cache["df"]


def history(ticker: str) -> pd.DataFrame:
    return data.series(prices(), ticker)


_ml_cache: dict = {"mtime": None, "data": None}


def ml() -> dict | None:
    """Resultados do motor de ML (data/ml/), recarregados quando o robô publica novos."""
    from pregao.ml import train  # import tardio: o app só precisa do ML na tela de Previsão

    marker = train.ML_DIR / "metrics.parquet"
    mtime = marker.stat().st_mtime if marker.exists() else None
    if _ml_cache["data"] is None or mtime != _ml_cache["mtime"]:
        _ml_cache["data"] = train.load()
        _ml_cache["mtime"] = mtime
    return _ml_cache["data"]
