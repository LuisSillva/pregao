"""Cálculos da tela de Comparação: vários ativos ao mesmo tempo.

Como na análise, nada aqui sabe de Dash: entram DataFrames, saem DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pregao import indicators as ind
from pregao.analysis import PERIODS as _ALL_PERIODS
from pregao.tickers import ASSETS, BY_TICKER

# Sem 6 meses: retorno anualizado precisa de pelo menos 1 ano (ver indicators.summary).
PERIODS = {k: v for k, v in _ALL_PERIODS.items() if k != "6M"}
MAX_ASSETS = 6  # mais que isso e as linhas viram um novelo
INDEXES = {a.ticker for a in ASSETS if a.sector == "Índice"}


def wide_prices(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Formato longo -> largo: uma coluna de adj_close por ticker, uma linha por data.

    Para comparar ativos lado a lado, o formato largo é o natural. As datas em que
    um mercado abriu e o outro não ficam com NaN naquela coluna.
    """
    subset = df[df["ticker"].isin(tickers)]
    wide = subset.pivot(index="date", columns="ticker", values="adj_close")
    return wide[[t for t in tickers if t in wide.columns]]  # mantém a ordem pedida


def cut(wide: pd.DataFrame, period: str) -> pd.DataFrame:
    offset = PERIODS[period]
    if offset is None:
        return wide
    return wide[wide.index >= wide.index.max() - offset]


def normalized(wide: pd.DataFrame) -> pd.DataFrame:
    """Cada coluna dividida pelo seu primeiro valor válido, × 100.

    Todo mundo parte de 100 no início do período, então o gráfico mostra quanto
    R$ 100 investidos em cada ativo teriam virado, independente do preço da ação.
    """
    return wide.apply(lambda col: col / col.dropna().iloc[0] * 100)


def risk_return(wide: pd.DataFrame) -> pd.DataFrame:
    """Volatilidade e retorno anualizado de cada coluna."""
    rows = []
    for ticker in wide.columns:
        s = ind.summary(wide[ticker])
        rows.append({"ticker": ticker, "volatility": s["volatility"], "cagr": s["cagr"]})
    return pd.DataFrame(rows).dropna()


def weekly_correlation(wide: pd.DataFrame) -> pd.DataFrame:
    """Correlação entre os retornos SEMANAIS (log) dos ativos.

    Por que semanal? A B3 fecha às 17h e Nova York às 18h (horário de Brasília),
    e os feriados não batem. Com retornos diários, uma notícia que sai às 17h30
    entra no "dia" do ativo americano mas só no dia seguinte do brasileiro, e a
    correlação entre mercados aparece menor do que é. Somando a semana inteira,
    esse descompasso quase some.
    """
    weekly = wide.resample("W-FRI").last()
    rets = np.log(weekly / weekly.shift(1))
    return rets.corr()  # o pandas ignora os NaN par a par


@dataclass
class Beta:
    benchmark: str
    beta: float  # inclinação: quanto a ação anda para cada 1% do índice
    alpha: float  # intercepto diário
    r2: float  # quanto do movimento da ação o índice explica (0 a 1)
    points: pd.DataFrame  # colunas "index" e "asset": retornos do mesmo dia


def benchmark_for(ticker: str) -> str:
    return "^BVSP" if BY_TICKER[ticker].market == "B3" else "^GSPC"


def beta(asset: pd.Series, index: pd.Series, benchmark: str) -> Beta:
    """Regressão linear simples: retorno da ação = alpha + beta × retorno do índice.

    Com uma variável só, a reta de mínimos quadrados tem fórmula fechada:
        beta  = cov(ação, índice) / var(índice)
        alpha = média(ação) - beta × média(índice)
    """
    points = pd.concat(
        {"index": ind.log_returns(index), "asset": ind.log_returns(asset)}, axis=1
    ).dropna()  # só os dias em que os dois negociaram
    x, y = points["index"], points["asset"]
    b = x.cov(y) / x.var()
    a = y.mean() - b * x.mean()
    r2 = x.corr(y) ** 2
    return Beta(benchmark=benchmark, beta=float(b), alpha=float(a), r2=float(r2), points=points)


def assign_colors(selected: list[str], previous: dict[str, int] | None, n_slots: int) -> dict[str, int]:
    """Dá a cada ativo selecionado um "slot" de cor e o mantém enquanto ele estiver selecionado.

    Se a cor fosse pela posição na lista, remover o 1º ativo repintaria todos os
    outros, e o usuário perderia a referência de qual linha é qual.
    Quem já tinha cor continua com ela; quem chega pega o primeiro slot livre.
    """
    previous = previous or {}
    kept = {t: previous[t] for t in selected if t in previous}
    free = [s for s in range(n_slots) if s not in kept.values()]
    for t in selected:
        if t not in kept:
            kept[t] = free.pop(0)
    return {t: kept[t] for t in selected}
