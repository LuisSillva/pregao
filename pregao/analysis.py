"""Prepara os dados de um ativo para a tela de análise.

Fica separado do Dash para poder ser testado sem subir o app.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pregao import indicators as ind

PERIODS = {
    "6M": pd.DateOffset(months=6),
    "1A": pd.DateOffset(years=1),
    "3A": pd.DateOffset(years=3),
    "5A": pd.DateOffset(years=5),
    "Máx": None,
}
MA_WINDOWS = [20, 50, 200]

# Tamanho de cada vela por período. Com 5 anos de velas diárias seriam ~1.250
# velas espremidas na tela; agrupando por semana (ou mês) o gráfico respira.
CANDLE_FREQ = {"6M": None, "1A": None, "3A": "W-FRI", "5A": "W-FRI", "Máx": "ME"}
CANDLE_LABEL = {None: "diárias", "W-FRI": "semanais", "ME": "mensais"}


@dataclass
class Analysis:
    view: pd.DataFrame  # dados diários do período
    chart: pd.DataFrame  # o mesmo período, agrupado no tamanho de vela do gráfico
    candle: str  # "diárias", "semanais" ou "mensais"
    stats: dict[str, float]  # retorno, volatilidade etc. do período
    drawdown: pd.Series
    last_close: float  # último preço de tela (sem ajuste)
    day_change: float  # variação do último dia
    last_date: pd.Timestamp


def adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Coloca abertura, máxima, mínima e fechamento na mesma escala do adj_close.

    O fator de ajuste de cada dia é adj_close / close. Aplicando o mesmo fator ao
    OHLC, as velas ficam coerentes com as médias móveis (que usam adj_close) e
    não aparece um "degrau" falso no dia de um desdobramento ou dividendo.
    """
    factor = df["adj_close"] / df["close"]
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = df[col] * factor
    return out


def resample(df: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    """Agrupa dias em semanas ou meses: cada vela nova resume as velas do grupo.

    Abertura = a do primeiro dia, máxima = a maior, mínima = a menor,
    fechamento = o do último dia, volume = a soma. Indicadores: o valor do último dia.
    """
    if freq is None:
        return df
    rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    rules |= {col: "last" for col in df.columns if col not in rules}
    out = df.resample(freq).agg(rules).dropna(subset=["close"])
    # o resample rotula cada grupo pelo FIM do período (sexta ou dia 30), que pode
    # estar no futuro; usamos o último dia que de fato teve pregão no grupo
    last_day = df["close"].resample(freq).apply(lambda s: s.index.max())
    out.index = pd.DatetimeIndex(last_day.loc[out.index], name=df.index.name)
    return out


def analyze(history: pd.DataFrame, period: str, ma_windows: list[int]) -> Analysis:
    """`history` é o histórico completo de um ticker, indexado por data."""
    adj = adjust_ohlc(history)
    price = adj["close"]

    # Indicadores calculados no histórico INTEIRO e só depois recortados.
    # Se recortássemos antes, a média de 200 dias de um gráfico de 6 meses
    # só começaria a aparecer no fim, por falta de dias anteriores.
    for window in ma_windows:
        adj[f"sma_{window}"] = ind.sma(price, window)
    adj["rsi"] = ind.rsi(price)

    offset = PERIODS[period]
    view = adj if offset is None else adj[adj.index >= adj.index[-1] - offset]

    freq = CANDLE_FREQ[period]
    return Analysis(
        view=view,
        chart=resample(view, freq),
        candle=CANDLE_LABEL[freq],
        stats=ind.summary(view["close"]),
        drawdown=ind.drawdown(view["close"]),
        last_close=float(history["close"].iloc[-1]),
        day_change=float(price.iloc[-1] / price.iloc[-2] - 1),
        last_date=history.index[-1],
    )
