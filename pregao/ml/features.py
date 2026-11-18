"""Transforma o histórico de preços em uma tabela para o modelo.

Cada linha é um dia `t`. As colunas de X (features) usam SÓ informação disponível
no fechamento de `t`. Os alvos olham para frente: o que acontece de t+1 até t+H.

A regra de ouro: nenhuma feature pode espiar o futuro. O teste
`test_features_nao_usam_o_futuro` muda os preços futuros e confere que as
features de hoje não mudam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pregao import indicators as ind

HORIZON = 5  # prevemos os próximos 5 pregões (uma semana)
ANNUAL = np.sqrt(ind.TRADING_DAYS)

FEATURES = [
    "ret_1", "ret_5", "ret_20", "ret_60",  # momento: quanto subiu/caiu recentemente
    "vol_5", "vol_20", "vol_60",  # risco recente, em várias janelas
    "rsi_14",
    "dist_sma_20", "dist_sma_200",  # quão longe o preço está das médias
    "drawdown_252",  # quão longe do topo do último ano
    "volume_z",  # volume de hoje comparado ao normal
    "idx_ret_5", "idx_ret_20", "idx_vol_20",  # o que o mercado como um todo está fazendo
]
TARGETS = ["retorno", "volatilidade"]


def build(history: pd.DataFrame, index_prices: pd.Series, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (X, y).

    `history`: histórico do ativo (adj_close, volume) indexado por data.
    `index_prices`: adj_close do índice de referência (Ibovespa ou S&P 500).
    `y` tem as colunas de TARGETS; nos últimos `horizon` dias ela é NaN,
    porque o futuro ainda não aconteceu. Essas são as linhas que vamos prever.
    """
    # Dias com volume zero não tiveram negócio: o Yahoo só repete o preço anterior.
    # A SUZB3, por exemplo, tem ~700 dias assim em 2015-2017, antes do ticker atual.
    # Mantê-los ensinaria ao modelo que a ação "ficou parada", o que não aconteceu.
    history = history[history["volume"] > 0]
    p = history["adj_close"]
    lr = ind.log_returns(p)
    X = pd.DataFrame(index=p.index)

    X["ret_1"] = lr
    for n in (5, 20, 60):
        X[f"ret_{n}"] = np.log(p / p.shift(n))
    for n in (5, 20, 60):
        X[f"vol_{n}"] = lr.rolling(n).std() * ANNUAL
    X["rsi_14"] = ind.rsi(p) / 100
    X["dist_sma_20"] = p / ind.sma(p, 20) - 1
    X["dist_sma_200"] = p / ind.sma(p, 200) - 1
    X["drawdown_252"] = p / p.rolling(252).max() - 1

    volume = history["volume"]
    X["volume_z"] = np.log(volume / volume.rolling(20).mean())

    # O índice tem seus próprios feriados. reindex coloca-o nas datas do ativo e
    # ffill repete o último valor conhecido: só olha para trás, então não vaza.
    idx = index_prices.reindex(p.index.union(index_prices.index)).ffill().reindex(p.index)
    idx_lr = np.log(idx / idx.shift(1))
    X["idx_ret_5"] = np.log(idx / idx.shift(5))
    X["idx_ret_20"] = np.log(idx / idx.shift(20))
    X["idx_vol_20"] = idx_lr.rolling(20).std() * ANNUAL

    y = pd.DataFrame(index=p.index)
    # retorno log acumulado de t até t+H
    y["retorno"] = np.log(p.shift(-horizon) / p)
    # volatilidade realizada nos dias t+1..t+H, anualizada.
    # rolling(H) no dia t+H cobre t+1..t+H; o shift(-H) traz esse valor para o dia t.
    # O piso de 0,1% evita log(0) nos modelos treinados no log do alvo.
    y["volatilidade"] = (lr.rolling(horizon).std().shift(-horizon) * ANNUAL).clip(lower=0.001)

    X = X[FEATURES].replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)  # descarta o primeiro ano, quando ainda não há médias de 200/252 dias
    return X[keep], y[keep]
