import numpy as np
import pandas as pd
import pytest

from pregao import analysis


def history(n=600, split_at=None):
    """Histórico falso. Com split_at, simula um desdobramento 1:2 naquele dia."""
    idx = pd.bdate_range("2023-01-02", periods=n)
    adj = pd.Series(np.linspace(50, 80, n), index=idx)
    close = adj.copy()
    if split_at is not None:
        close.iloc[:split_at] *= 2  # antes do desdobramento, o preço de tela era o dobro
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01, "low": close * 0.98,
        "close": close, "adj_close": adj, "volume": 1000,
    }, index=idx)


def test_adjust_ohlc_remove_degrau_do_desdobramento():
    adj = analysis.adjust_ohlc(history(100, split_at=50))
    jump = adj["close"].pct_change().abs().max()
    assert jump < 0.01
    assert (adj["high"] >= adj["close"]).all()


def test_media_longa_existe_no_periodo_curto():
    a = analysis.analyze(history(600), "6M", [200])
    assert a.view["sma_200"].notna().all()


def test_recorte_do_periodo():
    a = analysis.analyze(history(600), "1A", [20])
    span = a.view.index[-1] - a.view.index[0]
    assert pd.Timedelta(days=360) <= span <= pd.Timedelta(days=366)


def test_maximo_usa_tudo():
    assert len(analysis.analyze(history(600), "Máx", [20]).view) == 600


def test_ultimo_preco_e_o_de_tela():
    h = history(100, split_at=99)
    a = analysis.analyze(h, "Máx", [20])
    assert a.last_close == pytest.approx(h["close"].iloc[-1])
    assert abs(a.day_change) < 0.01  # o desdobramento não aparece como queda


def test_resample_semanal():
    h = analysis.adjust_ohlc(history(10))  # 2023-01-02 (seg) a 2023-01-13 (sex): 2 semanas
    w = analysis.resample(h, "W-FRI")
    assert len(w) == 2
    first_week = h.iloc[:5]
    assert w["open"].iloc[0] == first_week["open"].iloc[0]
    assert w["high"].iloc[0] == first_week["high"].max()
    assert w["close"].iloc[0] == first_week["close"].iloc[-1]
    assert w["volume"].iloc[0] == first_week["volume"].sum()


def test_resample_usa_ultimo_dia_com_pregao():
    h = analysis.adjust_ohlc(history(8))  # termina numa quarta-feira
    w = analysis.resample(h, "W-FRI")
    assert w.index[-1] == h.index[-1]


def test_periodos_longos_usam_velas_maiores():
    a = analysis.analyze(history(600), "Máx", [20])
    assert a.candle == "mensais"
    assert len(a.chart) < 40
