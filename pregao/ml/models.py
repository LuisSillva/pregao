"""Os concorrentes. Todos seguem a mesma interface do scikit-learn: fit(X, y) e predict(X).

Ter a mesma interface é o que permite o backtest tratar todos do mesmo jeito,
e comparar uma baseline de uma linha com um LightGBM de forma justa.
Herdar de BaseEstimator deixa nossos modelos simples "plugáveis" em ferramentas
do scikit-learn, como o TransformedTargetRegressor usado em on_log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Zero(BaseEstimator, RegressorMixin):
    """Baseline do retorno: "nos próximos dias, o preço não muda" (retorno 0)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class Persistence(BaseEstimator, RegressorMixin):
    """Baseline da volatilidade: "a próxima semana vai oscilar como o último mês"."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["vol_20"].to_numpy()


class HistoricalMean(BaseEstimator, RegressorMixin):
    """Prevê sempre a média do alvo no treino. Para retorno, é a tendência histórica."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def linear():
    """Regressão linear com regularização (Ridge).

    O StandardScaler coloca as features na mesma escala (média 0, desvio 1).
    Sem isso, a penalidade do Ridge trataria "volume_z" e "ret_1" de forma
    desigual só porque os números têm tamanhos diferentes.
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def lightgbm():
    """Gradient boosting: centenas de árvores pequenas, cada uma corrigindo o erro das anteriores.

    Os parâmetros são conservadores de propósito (árvores rasas, aprendizado lento,
    muitas amostras por folha), porque dados financeiros têm muito ruído e pouco
    sinal: um modelo solto decora o ruído do treino e vai mal no teste.
    """
    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def on_log(make_model):
    """Treina no log do alvo e devolve a previsão de volta na escala normal.

    A volatilidade é assimétrica: muitas semanas calmas e algumas explosivas.
    Modelos que minimizam o erro quadrático aprendem a MÉDIA, puxada pelas semanas
    explosivas, mas o MAE recompensa quem acerta a MEDIANA. No log, a distribuição
    fica quase simétrica, média e mediana se aproximam e o conflito some.
    """
    return lambda: TransformedTargetRegressor(regressor=make_model(), func=np.log, inverse_func=np.exp)


# Para cada alvo: nome do modelo -> função que cria um modelo novo (não treinado).
# O primeiro de cada lista é a baseline contra a qual os outros são medidos.
MODELS = {
    "retorno": {
        "Baseline (zero)": Zero,
        "Média histórica": HistoricalMean,
        "Regressão linear": linear,
        "LightGBM": lightgbm,
    },
    "volatilidade": {
        "Baseline (último mês)": Persistence,
        "Média histórica": on_log(HistoricalMean),
        "Regressão linear": on_log(linear),
        "LightGBM": on_log(lightgbm),
    },
}


def baseline_name(target: str) -> str:
    return next(iter(MODELS[target]))


def feature_importance(model, columns: list[str]) -> pd.Series | None:
    """Quanto cada feature foi usada pelo LightGBM (ganho total nas divisões)."""
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    if isinstance(model, LGBMRegressor):
        gain = model.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain / gain.sum(), index=columns).sort_values(ascending=False)
    return None
