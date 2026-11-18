import numpy as np
import pandas as pd
import pytest

from pregao.ml import backtest, features, metrics, models


def fake_history(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    price = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    index = 1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    hist = pd.DataFrame({"adj_close": price, "volume": rng.integers(1_000, 5_000, n)}, index=idx)
    return hist, pd.Series(index, index=idx)


# ---------------------------------------------------------------- features


def test_features_nao_usam_o_futuro():
    """Mexer nos preços DEPOIS do dia t não pode mudar nenhuma feature do dia t."""
    hist, idx = fake_history()
    X1, _ = features.build(hist, idx)
    t = X1.index[300]
    hist2, idx2 = hist.copy(), idx.copy()
    hist2.loc[hist2.index > t, "adj_close"] *= 3
    idx2.loc[idx2.index > t] *= 0.5
    X2, _ = features.build(hist2, idx2)
    pd.testing.assert_frame_equal(X1.loc[:t], X2.loc[:t])


def test_alvo_retorno_olha_h_dias_a_frente():
    hist, idx = fake_history()
    X, y = features.build(hist, idx, horizon=5)
    t = X.index[100]
    pos = hist.index.get_loc(t)
    expected = np.log(hist["adj_close"].iloc[pos + 5] / hist["adj_close"].iloc[pos])
    assert y.loc[t, "retorno"] == pytest.approx(expected)


def test_alvo_volatilidade_usa_os_proximos_h_retornos():
    hist, idx = fake_history()
    X, y = features.build(hist, idx, horizon=5)
    t = X.index[100]
    pos = hist.index.get_loc(t)
    lr = np.log(hist["adj_close"] / hist["adj_close"].shift(1))
    expected = lr.iloc[pos + 1: pos + 6].std() * np.sqrt(252)
    assert y.loc[t, "volatilidade"] == pytest.approx(expected)


def test_ultimos_dias_sem_alvo():
    hist, idx = fake_history()
    _, y = features.build(hist, idx, horizon=5)
    assert y["retorno"].iloc[-5:].isna().all()
    assert y["retorno"].iloc[:-5].notna().all()


def test_features_sem_nan():
    hist, idx = fake_history()
    X, _ = features.build(hist, idx)
    assert list(X.columns) == features.FEATURES
    assert X.notna().all().all()


# ---------------------------------------------------------------- backtest


def test_splits_respeitam_o_gap():
    """Toda linha de treino precisa ter o alvo conhecido no dia do teste: j + H <= início."""
    H = 5
    blocks = list(backtest.splits(1200, H, min_train=500, step=21))
    assert blocks
    for train_end, test_start, test_end in blocks:
        last_train_row = train_end - 1
        assert last_train_row + H <= test_start
        assert test_start < test_end


def test_splits_cobrem_tudo_sem_sobrepor():
    blocks = list(backtest.splits(1000, 5, min_train=500, step=21))
    covered = [i for _, a, b in blocks for i in range(a, b)]
    assert covered == list(range(blocks[0][1], 1000))


def test_walk_forward_nao_ve_o_futuro():
    """Um "modelo" que espia o maior índice do treino prova que nunca viu dados do teste."""
    seen = []

    class Spy(models.HistoricalMean):
        def fit(self, X, y):
            seen.append((X.index.max(), y.index.max()))
            return super().fit(X, y)

    hist, idx = fake_history()
    X, y = features.build(hist, idx)
    preds = backtest.walk_forward(X, y["retorno"], Spy, horizon=5, min_train=300, step=50)
    labeled = X.index[y["retorno"].notna()]
    for (max_x, _), test_date in zip(seen, preds.index[::50]):
        gap = labeled.get_loc(test_date) - labeled.get_loc(max_x)
        assert gap >= 5


# ---------------------------------------------------------------- modelos


def test_baselines():
    X = pd.DataFrame({"vol_20": [0.2, 0.3]})
    assert models.Zero().fit(X, None).predict(X).tolist() == [0, 0]
    assert models.Persistence().fit(X, None).predict(X).tolist() == [0.2, 0.3]


def test_media_no_log_e_geometrica():
    X = pd.DataFrame({"vol_20": [0, 0, 0]})
    m = models.on_log(models.HistoricalMean)().fit(X, np.array([0.1, 0.2, 0.4]))
    assert m.predict(X)[0] == pytest.approx(0.2)


def test_dias_sem_negocio_sao_descartados():
    hist, idx = fake_history()
    hist.iloc[400:420, hist.columns.get_loc("volume")] = 0
    X, y = features.build(hist, idx)
    assert not X.index.isin(hist.index[400:420]).any()
    assert (y["volatilidade"].dropna() > 0).all()
