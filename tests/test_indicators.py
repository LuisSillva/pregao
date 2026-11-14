import numpy as np
import pandas as pd
import pytest

from pregao import indicators as ind


def prices(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_returns():
    r = ind.returns(prices([100, 110, 99]))
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(-0.10)


def test_log_returns_somam():
    p = prices([100, 120, 90, 135])
    total_log = ind.log_returns(p).sum()
    assert np.exp(total_log) == pytest.approx(135 / 100)


def test_sma():
    s = ind.sma(prices([1, 2, 3, 4, 5]), 3)
    assert s.isna().sum() == 2
    assert s.iloc[-1] == pytest.approx(4.0)


def test_rsi_so_alta_e_100():
    r = ind.rsi(prices(range(1, 40)), 14)
    assert r.dropna().eq(100).all()


def test_rsi_so_queda_e_0():
    r = ind.rsi(prices(range(40, 1, -1)), 14)
    assert r.dropna().iloc[-1] == pytest.approx(0.0)


def test_rsi_entre_0_e_100():
    rng = np.random.default_rng(0)
    p = prices(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    r = ind.rsi(p).dropna()
    assert len(r) > 250
    assert r.between(0, 100).all()


def test_rsi_comeca_depois_da_janela():
    r = ind.rsi(prices(range(1, 40)), 14)
    assert r.iloc[:14].isna().all()


def test_volatilidade_anualizada():
    # Retorno log alternando +1% e -1%: desvio diário ~1%, anual ~ 1% * sqrt(252)
    logs = np.tile([0.01, -0.01], 50)
    p = prices(100 * np.exp(np.concatenate([[0], np.cumsum(logs)])))
    vol = ind.rolling_volatility(p, 20).iloc[-1]
    assert vol == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_drawdown():
    dd = ind.drawdown(prices([100, 120, 90, 130]))
    assert list(dd.round(4)) == [0.0, 0.0, -0.25, 0.0]


def test_summary():
    p = pd.Series([100.0, 150.0, 75.0, 121.0], index=pd.to_datetime(
        ["2020-01-01", "2020-06-01", "2021-01-01", "2022-01-01"]))
    s = ind.summary(p)
    assert s["total_return"] == pytest.approx(0.21)
    assert s["cagr"] == pytest.approx(0.10, abs=0.001)  # 1.1 ** 2 = 1.21
    assert s["max_drawdown"] == pytest.approx(-0.5)


def test_summary_poucos_dados():
    assert np.isnan(ind.summary(prices([100]))["total_return"])
