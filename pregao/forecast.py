"""Prepara os resultados do motor de ML (data/ml/) para a tela de Previsão.

Os números já foram calculados pelo robô de sábado; aqui só filtramos,
resumimos e traduzimos para algo que uma pessoa entenda.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pregao.ml import models

SIGNIFICANCE = 0.05

FEATURE_LABELS = {
    "ret_1": "retorno de 1 dia",
    "ret_5": "retorno de 5 dias",
    "ret_20": "retorno de 20 dias",
    "ret_60": "retorno de 60 dias",
    "vol_5": "volatilidade de 5 dias",
    "vol_20": "volatilidade de 20 dias",
    "vol_60": "volatilidade de 60 dias",
    "rsi_14": "RSI",
    "dist_sma_20": "distância da MM 20",
    "dist_sma_200": "distância da MM 200",
    "drawdown_252": "distância do topo de 1 ano",
    "volume_z": "volume relativo",
    "idx_ret_5": "índice: retorno de 5 dias",
    "idx_ret_20": "índice: retorno de 20 dias",
    "idx_vol_20": "índice: volatilidade de 20 dias",
}


def challengers(target: str) -> list[str]:
    """Os modelos que competem com a baseline, na ordem fixa (a cor segue essa ordem)."""
    return list(models.MODELS[target])[1:]


def verdict(skill: float, p_value: float) -> tuple[str, str]:
    """Traduz skill + p-valor numa frase e numa classe de estilo ('good', 'bad', 'neutral')."""
    if np.isnan(p_value):
        return "referência", "neutral"
    if p_value < SIGNIFICANCE:
        return ("melhor que a baseline", "good") if skill > 0 else ("pior que a baseline", "bad")
    return "empate (diferença pode ser sorte)", "neutral"


def cumulative_edge(bt: pd.DataFrame, target: str) -> pd.DataFrame:
    """Vantagem acumulada de cada modelo sobre a baseline, ao longo do tempo.

    Em cada data: |erro da baseline| - |erro do modelo|. Positivo = o modelo errou
    menos naquele dia. Somando ao longo do tempo, a linha SOBE enquanto o modelo
    está ganhando e DESCE enquanto está perdendo. Uma linha que sobe até 2020 e
    depois cai mostra um modelo que parou de funcionar.
    """
    wide = bt.pivot(index="date", columns="model", values="y_pred")
    y_true = bt.drop_duplicates("date").set_index("date")["y_true"]
    base_err = (y_true - wide[models.baseline_name(target)]).abs()
    out = pd.DataFrame(index=wide.index)
    for name in challengers(target):
        out[name] = (base_err - (y_true - wide[name]).abs()).cumsum()
    return out


def overview(metrics: pd.DataFrame, target: str) -> pd.DataFrame:
    """Skill de todos os modelos em todas as ações, para o gráfico de resumo."""
    m = metrics[(metrics["target"] == target) & (~metrics["is_baseline"])].copy()
    m["significant"] = m["p_value"] < SIGNIFICANCE
    return m[["ticker", "model", "skill", "p_value", "significant"]]
