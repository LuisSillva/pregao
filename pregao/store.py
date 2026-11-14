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
