"""Como medir se um modelo presta.

Uma métrica sozinha engana. Por isso olhamos várias, sempre em comparação
com a baseline, e no fim perguntamos se a diferença não foi sorte.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y_true, y_pred) -> float:
    """Erro absoluto médio: em média, quanto a previsão errou (na unidade do alvo)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Raiz do erro quadrático médio: como o MAE, mas pune mais os erros grandes."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def skill(model_mae: float, baseline_mae: float) -> float:
    """1 - erro do modelo / erro da baseline.

    0 = igual à baseline. 0,10 = erra 10% menos. Negativo = pior que a baseline.
    """
    return 1 - model_mae / baseline_mae


def direction_accuracy(y_true, y_pred) -> float:
    """Em que fração dos casos o modelo acertou se ia subir ou cair."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_pred != 0  # previsão exatamente zero não aposta em direção nenhuma
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def up_rate(y_true) -> float:
    """Fração de períodos em que o ativo subiu.

    Se subiu em 55% das semanas, quem diz "sempre sobe" acerta a direção 55% das
    vezes sem saber nada. Acerto de direção só impressiona acima disso.
    """
    return float(np.mean(np.asarray(y_true) > 0))


def diebold_mariano(y_true, pred_a, pred_b, horizon: int) -> tuple[float, float]:
    """Teste de Diebold-Mariano (com a correção de Harvey, Leybourne e Newbold).

    Pergunta: a diferença de erro entre os modelos A e B é real ou pode ser sorte?
    Devolve (estatística, p-valor). Estatística negativa = A erra menos que B.
    p-valor < 0,05 costuma ser lido como "a diferença é significativa".

    Passo a passo:
    1. d_t = |erro de A| - |erro de B| em cada dia.
    2. Se os modelos fossem igualmente bons, a média de d seria 0.
    3. Os alvos se sobrepõem (o retorno de 5 dias de hoje e o de amanhã
       compartilham 4 dias), então os d_t são correlacionados entre si. A variância
       precisa considerar isso: somamos as autocovariâncias até a defasagem H-1
       (estimador de Newey-West).
    4. Estatística = média / erro padrão, comparada com uma distribuição t.
    """
    y_true = np.asarray(y_true)
    d = np.abs(y_true - np.asarray(pred_a)) - np.abs(y_true - np.asarray(pred_b))
    n = len(d)
    d_mean = d.mean()
    centered = d - d_mean
    var = np.sum(centered**2) / n
    for lag in range(1, horizon):
        cov = np.sum(centered[lag:] * centered[:-lag]) / n
        var += 2 * cov
    if var <= 0:
        return float("nan"), float("nan")
    dm = d_mean / np.sqrt(var / n)
    # correção para amostras finitas
    h = horizon
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)
