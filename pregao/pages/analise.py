"""Tela de análise de um ativo."""

from __future__ import annotations

import math

import dash
from dash import Input, Output, callback, dcc, html

from pregao import analysis, charts, store, theme
from pregao.tickers import ASSETS, BY_TICKER

dash.register_page(__name__, path="/", name="Análise", order=0)

DEFAULT_TICKER = "PETR4.SA"
# Cor fixa por janela: a cor segue a média, não a posição dela na lista.
MA_COLORS = dict(zip(analysis.MA_WINDOWS, theme.SERIES))
GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


def ticker_label(ticker: str) -> str:
    asset = BY_TICKER[ticker]
    code = ticker.removesuffix(".SA")
    return f"{code} · {asset.name}"


layout = html.Div(
    className="page",
    children=[
        # Linha de filtros: tudo que muda o que aparece na tela fica junto, no topo
        html.Div(
            className="filters",
            children=[
                dcc.Dropdown(
                    id="ticker",
                    options=[{"label": ticker_label(a.ticker), "value": a.ticker} for a in ASSETS],
                    value=DEFAULT_TICKER,
                    clearable=False,
                    className="ticker-select",
                ),
                dcc.RadioItems(
                    id="period",
                    options=list(analysis.PERIODS),
                    value="1A",
                    className="segmented",
                    inline=True,
                ),
                dcc.Checklist(
                    id="mas",
                    options=[{"label": f"MM {w}", "value": w} for w in analysis.MA_WINDOWS],
                    value=[20, 200],
                    className="segmented",
                    inline=True,
                ),
            ],
        ),
        html.Div(id="headline", className="headline"),
        html.Div(id="stats", className="stats"),
        html.Section(
            className="panel",
            children=[dcc.Graph(id="price-chart", config=GRAPH_CONFIG, style={"height": "560px"})],
        ),
        html.Section(
            className="panel",
            children=[
                html.H2("Distância do topo (drawdown)", className="panel-title"),
                html.P(
                    "Quanto o preço está abaixo do maior valor já atingido no período.",
                    className="panel-note",
                ),
                dcc.Graph(id="drawdown-chart", config=GRAPH_CONFIG, style={"height": "220px"}),
            ],
        ),
    ],
)


def pct(value: float, signed: bool = True) -> str:
    if value is None or math.isnan(value):
        return "–"
    text = f"{value:+.1%}" if signed else f"{value:.1%}"
    return text.replace(".", ",")


def money(value: float, currency: str) -> str:
    symbol = "R$" if currency == "BRL" else "US$"
    # formata no padrão brasileiro: 1.234,56
    text = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{symbol} {text}"


def stat_tile(label: str, value: str, hint: str) -> html.Div:
    return html.Div(
        className="stat",
        children=[
            html.Div(label, className="stat-label"),
            html.Div(value, className="stat-value"),
            html.Div(hint, className="stat-hint"),
        ],
    )


@callback(
    Output("headline", "children"),
    Output("stats", "children"),
    Output("price-chart", "figure"),
    Output("drawdown-chart", "figure"),
    Input("ticker", "value"),
    Input("period", "value"),
    Input("mas", "value"),
)
def update(ticker: str, period: str, mas: list[int]):
    asset = BY_TICKER[ticker]
    a = analysis.analyze(store.history(ticker), period, analysis.MA_WINDOWS)

    up = a.day_change >= 0
    headline = [
        html.Div([html.H1(ticker_label(ticker)), html.Span(f"{asset.market} · {asset.sector}", className="muted")]),
        html.Div(
            className="quote",
            children=[
                html.Span(money(a.last_close, asset.currency), className="price"),
                html.Span(
                    f"{'▲' if up else '▼'} {pct(a.day_change)}",
                    className="delta up" if up else "delta down",
                ),
                html.Span(f"fechamento de {a.last_date:%d/%m/%Y} · velas {a.candle}", className="muted"),
            ],
        ),
    ]

    s = a.stats
    stats = [
        stat_tile("Retorno no período", pct(s["total_return"]), "com dividendos reinvestidos"),
        stat_tile(
            "Retorno ao ano",
            pct(s["cagr"]),
            "só a partir de 1 ano de dados" if math.isnan(s["cagr"]) else "taxa anual equivalente",
        ),
        stat_tile("Volatilidade", pct(s["volatility"], signed=False), "anualizada"),
        stat_tile("Pior queda", pct(s["max_drawdown"]), "do topo ao fundo no período"),
    ]

    price_fig = charts.price_figure(
        a.chart, {w: MA_COLORS[w] for w in sorted(mas)}, daily=a.candle == "diárias"
    )
    dd_fig = charts.drawdown_figure(a.drawdown)
    return headline, stats, price_fig, dd_fig
