"""Tela de previsão: o que os modelos dizem para a próxima semana, e quanto confiar neles."""

from __future__ import annotations

import math
from datetime import datetime

import dash
from dash import Input, Output, callback, dcc, html

from pregao import charts, forecast, store, theme
from pregao.ml import features, models
from pregao.tickers import ASSETS
from pregao.ui import GRAPH_CONFIG, ticker_label

dash.register_page(__name__, path="/previsao", name="Previsão", order=2)

STOCKS = [a.ticker for a in ASSETS if a.sector != "Índice"]
TARGET_LABELS = {"retorno": "Retorno", "volatilidade": "Volatilidade"}
TARGET_NOTES = {
    "retorno": "Retorno acumulado nos próximos 5 pregões. A baseline diz que o preço não muda (retorno zero).",
    "volatilidade": "Quanto a ação deve oscilar nos próximos 5 pregões, em escala anual. "
                    "A baseline diz que vai oscilar como no último mês.",
}


def model_colors(target: str) -> dict[str, str]:
    """Cor fixa por modelo, na ordem validada da paleta. A baseline fica em cinza."""
    return dict(zip(forecast.challengers(target), theme.CATEGORICAL))


def fmt(value: float, target: str, signed: bool | None = None) -> str:
    if value is None or math.isnan(value):
        return "–"
    signed = target == "retorno" if signed is None else signed
    return (f"{value:+.1%}" if signed else f"{value:.1%}").replace(".", ",")


def panel(title: str, note: str, body, extra=None) -> html.Section:
    head = [html.H2(title, className="panel-title"), html.P(note, className="panel-note")]
    return html.Section(className="panel", children=[
        html.Div([html.Div(head), extra] if extra is not None else head, className="panel-head"), body,
    ])


layout = html.Div(
    className="page",
    children=[
        html.Div(
            className="filters",
            children=[
                dcc.Dropdown(
                    id="fc-ticker",
                    options=[{"label": ticker_label(t), "value": t} for t in STOCKS],
                    value="PETR4.SA", clearable=False, className="ticker-select",
                ),
                dcc.RadioItems(
                    id="fc-target",
                    options=[{"label": v, "value": k} for k, v in TARGET_LABELS.items()],
                    value="retorno", className="segmented", inline=True,
                ),
            ],
        ),
        html.Div(
            className="disclaimer",
            children="Projeto de estudo. Nada aqui é recomendação de investimento: os próprios números "
                     "abaixo mostram que prever a direção do preço não funciona.",
        ),
        html.Div(id="fc-headline", className="fc-headline"),
        html.Div(id="fc-tiles", className="fc-tiles"),
        html.Section(className="panel", children=[
            html.H2("Como cada modelo se saiu no passado", className="panel-title"),
            html.P(id="fc-table-note", className="panel-note"),
            html.Div(id="fc-table", className="table-wrap"),
        ]),
        html.Div(
            className="grid-2",
            children=[
                panel(
                    "Previsto × realizado",
                    "Cada ponto é uma semana do backtest. Um modelo perfeito poria todos na diagonal; "
                    "uma nuvem redonda significa que a previsão não diz nada.",
                    dcc.Graph(id="fc-scatter", config=GRAPH_CONFIG, style={"height": "380px"}),
                    extra=dcc.RadioItems(id="fc-model", className="segmented", inline=True),
                ),
                panel(
                    "Vantagem acumulada sobre a baseline",
                    "A linha sobe nas semanas em que o modelo errou menos que a baseline e desce nas "
                    "semanas em que errou mais.",
                    dcc.Graph(id="fc-edge", config=GRAPH_CONFIG, style={"height": "380px"}),
                ),
            ],
        ),
        html.Div(
            className="grid-2",
            children=[
                panel(
                    "O que o LightGBM mais usou",
                    "Participação de cada feature nas decisões das árvores, no modelo treinado hoje.",
                    dcc.Graph(id="fc-importance", config=GRAPH_CONFIG, style={"height": "420px"}),
                ),
                panel(
                    "Todas as ações",
                    "Skill de cada modelo nas 17 ações. Ponto cheio: diferença estatisticamente "
                    "significativa (p < 0,05). Ponto vazado: pode ter sido sorte.",
                    dcc.Graph(id="fc-overview", config=GRAPH_CONFIG, style={"height": "420px"}),
                ),
            ],
        ),
        html.P(id="fc-meta", className="muted small"),
    ],
)


def metrics_table(m, target: str) -> html.Table:
    is_return = target == "retorno"
    header = ["Modelo", "Erro médio", "Skill"] + (["Acerto de direção"] if is_return else []) + ["p-valor", "Veredito"]
    rows = []
    for _, r in m.iterrows():
        text, kind = forecast.verdict(r["skill"], r["p_value"])
        cells = [
            html.Td(r["model"], className="name"),
            html.Td(fmt(r["mae"], target, signed=False), className="num"),
            html.Td("–" if r["is_baseline"] else fmt(r["skill"], target, signed=True), className="num"),
        ]
        if is_return:
            cells.append(html.Td(fmt(r["direction"], target, signed=False), className="num"))
        cells += [
            html.Td("–" if math.isnan(r["p_value"]) else f"{r['p_value']:.3f}".replace(".", ","), className="num"),
            html.Td(html.Span(text, className=f"chip {kind}")),
        ]
        rows.append(html.Tr(cells, className="baseline" if r["is_baseline"] else None))
    return html.Table(className="metrics", children=[
        html.Thead(html.Tr([html.Th(h) for h in header])), html.Tbody(rows),
    ])


@callback(
    Output("fc-model", "options"),
    Output("fc-model", "value"),
    Input("fc-target", "value"),
)
def model_options(target: str):
    names = list(models.MODELS[target])
    return [{"label": n.split(" (")[0], "value": n} for n in names], names[-1]


@callback(
    Output("fc-headline", "children"),
    Output("fc-tiles", "children"),
    Output("fc-table-note", "children"),
    Output("fc-table", "children"),
    Output("fc-edge", "figure"),
    Output("fc-importance", "figure"),
    Output("fc-overview", "figure"),
    Output("fc-meta", "children"),
    Input("fc-ticker", "value"),
    Input("fc-target", "value"),
)
def update(ticker: str, target: str):
    res = store.ml()
    if res is None:
        empty = html.P("Os modelos ainda não foram treinados. Rode scripts/train_models.py.")
        return empty, None, None, None, {}, {}, {}, None

    colors = model_colors(target)
    metrics = res["metrics"].query("ticker == @ticker and target == @target")
    latest = res["latest"].query("ticker == @ticker and target == @target").set_index("model")
    bt = res["backtest"].query("ticker == @ticker and target == @target")
    as_of = latest["as_of"].iloc[0]

    headline = [
        html.H1(f"{ticker_label(ticker)} · {TARGET_LABELS[target].lower()} nos próximos 5 pregões"),
        html.P([TARGET_NOTES[target], f" Previsões feitas com o fechamento de {as_of:%d/%m/%Y}."],
               className="muted"),
    ]

    by_model = metrics.set_index("model")
    tiles = []
    for name, row in latest.iterrows():
        text, kind = forecast.verdict(by_model.loc[name, "skill"], by_model.loc[name, "p_value"])
        swatch = theme.MUTED if name not in colors else colors[name]
        tiles.append(html.Div(className="stat fc-tile", children=[
            html.Div([html.Span(className="swatch", style={"background": swatch}), name], className="stat-label"),
            html.Div(fmt(row["prediction"], target), className="stat-value"),
            html.Div(f"80% de chance entre {fmt(row['low'], target)} e {fmt(row['high'], target)}",
                     className="stat-hint"),
            html.Span(text, className=f"chip {kind}"),
        ]))

    start, end = metrics["start"].min(), metrics["end"].max()
    table_note = (
        f"Backtest walk-forward de {start:%m/%Y} a {end:%m/%Y} ({int(metrics['n'].iloc[0])} previsões): "
        "em cada mês, o modelo foi treinado só com o passado e previu o mês seguinte. "
        "Skill = quanto o modelo errou menos que a baseline."
    )
    if target == "retorno":
        up = metrics["up_rate"].iloc[0]
        table_note += (f" A ação subiu em {fmt(up, target, signed=False)} das semanas, então um acerto de "
                       "direção abaixo disso perde para quem sempre aposta em alta.")

    imp = res["importance"].query("ticker == @ticker and target == @target")
    imp = imp.set_index("feature")["gain"].rename(index=forecast.FEATURE_LABELS)

    trained = datetime.fromisoformat(res["meta"]["trained_at"])
    meta = (f"Modelos treinados em {trained:%d/%m/%Y}. Retreinados todo sábado. "
            f"Horizonte de {res['meta']['horizon']} pregões; retreino a cada {res['meta']['retrain_every']} "
            f"pregões no backtest; intervalo de 80% a partir dos erros dos últimos 2 anos.")

    return (
        headline,
        tiles,
        table_note,
        metrics_table(metrics, target),
        charts.edge_figure(forecast.cumulative_edge(bt, target), colors),
        charts.importance_figure(imp, colors["LightGBM"]),
        charts.overview_figure(forecast.overview(res["metrics"], target), colors),
        meta,
    )


@callback(
    Output("fc-scatter", "figure"),
    Input("fc-ticker", "value"),
    Input("fc-target", "value"),
    Input("fc-model", "value"),
)
def update_scatter(ticker: str, target: str, model: str):
    res = store.ml()
    if res is None or model not in models.MODELS[target]:
        return {}
    bt = res["backtest"].query("ticker == @ticker and target == @target and model == @model")
    color = model_colors(target).get(model, theme.INK_2)
    return charts.prediction_scatter(bt, color, model, log=target == "volatilidade")
