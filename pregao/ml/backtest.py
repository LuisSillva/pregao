"""Backtest walk-forward: simula como o modelo teria se saído se estivesse rodando de verdade.

    treino: [=========]              teste: [---]
    treino: [============]           teste:     [---]
    treino: [===============]        teste:         [---]

A cada passo o modelo é treinado só com o passado e prevê o bloco seguinte.
Depois a janela anda e ele é treinado de novo, agora com mais dados.

Detalhe que derruba muitos backtests: o alvo do dia t usa preços até t+H.
Então, no dia em que vamos prever, as últimas H linhas do treino ainda não
"aconteceram" por completo. Usá-las seria espiar o futuro. Por isso existe
o `gap` (em inglês, purge) entre o fim do treino e o início do teste.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd

MIN_TRAIN = 756  # 3 anos de pregões antes da primeira previsão
STEP = 21  # retreina a cada ~1 mês


def splits(n: int, horizon: int, min_train: int = MIN_TRAIN, step: int = STEP) -> Iterator[tuple[int, int, int]]:
    """Gera (fim_do_treino, inicio_do_teste, fim_do_teste) como posições de linha.

    Treino = linhas [0, fim_do_treino). Teste = linhas [inicio, fim).
    Uma linha j do treino tem o alvo conhecido no dia `inicio` se j + horizon <= inicio,
    então o treino vai até inicio - horizon (inclusive).
    """
    start = min_train + horizon
    for test_start in range(start, n, step):
        train_end = test_start - horizon + 1
        yield train_end, test_start, min(test_start + step, n)


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    make_model: Callable,
    horizon: int,
    min_train: int = MIN_TRAIN,
    step: int = STEP,
) -> pd.Series:
    """Previsões fora da amostra para todas as linhas com alvo conhecido."""
    labeled = y.notna().to_numpy()
    X, y = X[labeled], y[labeled]
    preds = pd.Series(np.nan, index=X.index)
    for train_end, test_start, test_end in splits(len(X), horizon, min_train, step):
        model = make_model().fit(X.iloc[:train_end], y.iloc[:train_end])
        preds.iloc[test_start:test_end] = model.predict(X.iloc[test_start:test_end])
    return preds.dropna()


def fit_latest(X: pd.DataFrame, y: pd.Series, make_model: Callable):
    """Treina com tudo que já tem alvo conhecido e prevê a última linha (o dia de hoje)."""
    labeled = y.notna()
    model = make_model().fit(X[labeled], y[labeled])
    return model, float(model.predict(X.iloc[[-1]])[0])
