"""Indicadores técnicos e estatísticas de risco.

Todas as funções recebem uma Series de preços indexada por data e devolvem uma
Series do mesmo tamanho (ou um número). São funções puras: não leem arquivo, não
desenham nada. Isso deixa fácil testar e reaproveitar, inclusive como features
dos modelos de previsão mais à frente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # dias úteis de pregão num ano, usado para anualizar


def returns(prices: pd.Series) -> pd.Series:
    """Retorno simples dia a dia: (hoje / ontem) - 1. O primeiro dia é NaN."""
    return prices.pct_change()


def log_returns(prices: pd.Series) -> pd.Series:
    """Retorno logarítmico: ln(hoje / ontem).

    Para variações pequenas é quase igual ao simples, mas tem uma vantagem:
    retornos log se SOMAM ao longo do tempo (os simples se multiplicam).
    Por isso são os preferidos em estatística e nos modelos.
    """
    return np.log(prices / prices.shift(1))


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Média móvel simples: média dos últimos `window` preços."""
    return prices.rolling(window).mean()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Índice de Força Relativa (RSI), de 0 a 100, pela fórmula original de Wilder.

    Compara o tamanho médio das altas com o das quedas recentes.
    Acima de 70 costuma ser lido como "sobrecomprado"; abaixo de 30, "sobrevendido".
    """
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    # Wilder usa uma média exponencial com alpha = 1/window.
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # Sem nenhuma queda no período, avg_loss = 0 e o RSI é 100 por definição.
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def rolling_volatility(prices: pd.Series, window: int = 21) -> pd.Series:
    """Volatilidade anualizada: desvio padrão dos retornos log na janela × √252.

    Multiplica por √252 (e não por 252) porque a variância cresce linearmente com
    o tempo, então o desvio padrão cresce com a raiz quadrada.
    """
    return log_returns(prices).rolling(window).std() * np.sqrt(TRADING_DAYS)


def drawdown(prices: pd.Series) -> pd.Series:
    """Quanto o preço está abaixo do maior valor já atingido até aquele dia (≤ 0)."""
    peak = prices.cummax()
    return prices / peak - 1


def summary(prices: pd.Series) -> dict[str, float]:
    """Estatísticas do período: retorno total, retorno anualizado, volatilidade e pior queda."""
    prices = prices.dropna()
    if len(prices) < 2:
        return {"total_return": np.nan, "cagr": np.nan, "volatility": np.nan, "max_drawdown": np.nan}

    total = prices.iloc[-1] / prices.iloc[0] - 1
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = log_returns(prices).std() * np.sqrt(TRADING_DAYS)
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "volatility": float(vol),
        "max_drawdown": float(drawdown(prices).min()),
    }
