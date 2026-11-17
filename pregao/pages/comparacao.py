"""Tela de comparação entre ativos."""

from __future__ import annotations

import dash
from dash import Input, Output, State, callback, dcc, html

from pregao import charts, comparison, store, theme
from pregao.ui import GRAPH_CONFIG, ticker_label
from pregao.tickers import ASSETS

dash.register_page(__name__, path="/comparacao", name="Comparação", order=1)

DEFAULT = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "^BVSP"]
STOCKS = [a for a in ASSETS if a.ticker not in comparison.INDEXES]


def panel(title: str, note: str, graph_id: str, height: int, extra=None) -> html.Section:
    header = [html.H2(title, className="panel-title"), html.P(note, className="panel-note")]
    return html.Section(
        className="panel",
        children=[
            html.Div([html.Div(header), extra] if extra else header, className="panel-head"),
            dcc.Graph(id=graph_id, config=GRAPH_CONFIG, style={"height": f"{height}px"}),
        ],
    )


layout = html.Div(
    className="page",
    children=[
        # dcc.Store guarda dados no navegador entre um callback e outro.
        # Aqui: qual slot de cor cada ativo selecionado está usando.
        dcc.Store(id="color-slots", data={}),
        html.Div(
            className="filters",
            children=[
                dcc.Dropdown(
                    id="cmp-tickers",
                    options=[{"label": ticker_label(a.ticker), "value": a.ticker} for a in ASSETS],
                    value=DEFAULT,
                    multi=True,
                    clearable=False,
                    className="ticker-multi",
                ),
                dcc.RadioItems(
                    id="cmp-period", options=list(comparison.PERIODS), value="5A",
                    className="segmented", inline=True,
                ),
            ],
        ),
        html.Div(id="cmp-note", className="muted"),
        panel(
            "Quanto R$ 100 teriam virado",
            "Preços ajustados por dividendos; todos partem de 100 no início do período. "
            "Escala logarítmica: a mesma distância vertical é a mesma variação percentual. Índices tracejados.",
            "cmp-normalized", 420,
        ),
        html.Div(
            className="grid-2",
            children=[
                panel(
                    "Risco × retorno",
                    "Cada ponto é um ativo no período. Acima e à esquerda: mais retorno com menos oscilação.",
                    "cmp-risk-return", 380,
                ),
                panel(
                    "Correlação",
                    "Entre os retornos semanais. Perto de 1: andam juntos; perto de 0: independentes.",
                    "cmp-corr", 380,
                ),
            ],
        ),
        panel(
            "Ação × índice (beta)",
            "Cada ponto é um dia. A inclinação da reta é o beta: quanto a ação tende a andar "
            "para cada 1% do índice do seu mercado.",
            "cmp-beta", 420,
            extra=html.Div(
                className="beta-controls",
                children=[
                    dcc.Dropdown(
                        id="beta-ticker",
                        options=[{"label": ticker_label(a.ticker), "value": a.ticker} for a in STOCKS],
                        value="PETR4.SA", clearable=False, className="ticker-select",
                    ),
                    html.Div(id="beta-stats", className="beta-stats"),
                ],
            ),
        ),
    ],
)


@callback(
    Output("cmp-tickers", "value"),
    Output("color-slots", "data"),
    Output("cmp-note", "children"),
    Input("cmp-tickers", "value"),
    State("color-slots", "data"),  # State: lê o valor sem disparar o callback quando ele muda
)
def select(tickers: list[str], slots: dict):
    """Limita a seleção e decide a cor de cada ativo, sem repintar quem já estava lá."""
    note = None
    if len(tickers) > comparison.MAX_ASSETS:
        tickers = tickers[: comparison.MAX_ASSETS]
        note = f"Máximo de {comparison.MAX_ASSETS} ativos ao mesmo tempo."
    if not tickers:
        tickers = DEFAULT[:1]
    return tickers, comparison.assign_colors(tickers, slots, len(theme.CATEGORICAL)), note


@callback(
    Output("cmp-normalized", "figure"),
    Output("cmp-risk-return", "figure"),
    Output("cmp-corr", "figure"),
    Input("color-slots", "data"),
    Input("cmp-period", "value"),
)
def update(slots: dict, period: str):
    tickers = list(slots)
    colors = {t: theme.CATEGORICAL[i] for t, i in slots.items()}
    prices = store.prices()

    wide = comparison.cut(comparison.wide_prices(prices, tickers), period)
    every = comparison.cut(comparison.wide_prices(prices, [a.ticker for a in ASSETS]), period)

    return (
        charts.normalized_figure(comparison.normalized(wide), colors, comparison.INDEXES),
        charts.risk_return_figure(comparison.risk_return(every), colors),
        charts.correlation_figure(comparison.weekly_correlation(wide)),
    )


@callback(
    Output("cmp-beta", "figure"),
    Output("beta-stats", "children"),
    Input("beta-ticker", "value"),
    Input("cmp-period", "value"),
)
def update_beta(ticker: str, period: str):
    bench = comparison.benchmark_for(ticker)
    wide = comparison.cut(comparison.wide_prices(store.prices(), [ticker, bench]), period)
    b = comparison.beta(wide[ticker], wide[bench], bench)
    names = (charts.short(bench).replace("^BVSP", "Ibovespa").replace("^GSPC", "S&P 500"), charts.short(ticker))
    fig = charts.beta_figure(b.points, b.beta, b.alpha, theme.CATEGORICAL[0], names)
    stats = [
        html.Span([html.B(f"β {b.beta:.2f}".replace(".", ",")), f" contra o {names[0]}"]),
        html.Span(f"R² {b.r2:.2f}".replace(".", ","), title="quanto do movimento diário o índice explica"),
        html.Span(f"{len(b.points)} dias", className="muted"),
    ]
    return fig, stats
