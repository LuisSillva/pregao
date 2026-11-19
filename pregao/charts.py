"""Monta as figuras Plotly da tela de análise.

Cada função recebe dados prontos e devolve um `go.Figure`. Nenhuma delas sabe
nada de Dash: dá para testar e até usar num notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pregao import theme


def missing_business_days(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Dias úteis sem pregão (feriados) entre a primeira e a última data.

    O Plotly desenha o eixo de datas contínuo, então fins de semana e feriados
    viram buracos no gráfico de velas. Passamos esses dias para `rangebreaks`
    e o eixo os pula.
    """
    all_days = pd.bdate_range(index.min(), index.max())
    return list(all_days.difference(index))


def price_figure(df: pd.DataFrame, ma_colors: dict[int, str], daily: bool = True) -> go.Figure:
    """Velas + médias móveis, volume e RSI, empilhados com o mesmo eixo de datas.

    Espera um DataFrame com open/high/low/close/volume já ajustados, `rsi` e
    colunas `sma_<n>` para cada janela em `ma_colors` ({janela: cor}).
    A cor vem de fora e é fixa por janela: a MM 200 é sempre da mesma cor,
    mesmo se a MM 20 for desligada.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.62, 0.14, 0.24],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Preço",
            hovertemplate=(
                "abertura %{open:.2f}<br>máxima %{high:.2f}<br>"
                "mínima %{low:.2f}<br>fechamento %{close:.2f}<extra></extra>"
            ),
            increasing=dict(line=dict(color=theme.INK_2, width=1), fillcolor=theme.SURFACE),
            decreasing=dict(line=dict(color=theme.MUTED, width=1), fillcolor=theme.MUTED),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for window, color in ma_colors.items():
        col = f"sma_{window}"
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=f"MM {window}",
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate="%{y:.2f}",
            ),
            row=1,
            col=1,
        )
        last = df[col].dropna()
        if not last.empty:  # rótulo direto no fim da linha, além da legenda
            fig.add_annotation(
                x=last.index[-1], y=last.iloc[-1], text=f"MM {window}", showarrow=False,
                xanchor="left", xshift=6, font=dict(size=11, color=theme.INK_2), row=1, col=1,
            )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker=dict(color=theme.AXIS, line=dict(width=0)),
            showlegend=False,
            hovertemplate="%{y:.3s}",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["rsi"],
            name="RSI 14",
            mode="lines",
            line=dict(color=theme.INK_2, width=1.5),
            showlegend=False,
            hovertemplate="%{y:.0f}",
        ),
        row=3,
        col=1,
    )
    for level in (30, 70):
        fig.add_hline(y=level, line=dict(color=theme.MUTED, width=1, dash="dot"), row=3, col=1)

    fig.update_layout(**theme.base_layout(xaxis_rangeslider_visible=False, bargap=0.2))
    theme.style_axes(fig)
    fig.update_xaxes(hoverformat="%d/%m/%Y")
    if daily:  # com velas semanais/mensais não há buracos a esconder
        fig.update_xaxes(rangebreaks=[dict(values=missing_business_days(df.index)), dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Volume", title_font=dict(size=11, color=theme.MUTED), row=2, col=1)
    fig.update_yaxes(title_text="RSI", title_font=dict(size=11, color=theme.MUTED), range=[0, 100],
                     tickvals=[30, 70], row=3, col=1)
    return fig


def drawdown_figure(dd: pd.Series) -> go.Figure:
    """Área abaixo de zero mostrando a distância até o topo anterior."""
    fig = go.Figure(
        go.Scatter(
            x=dd.index,
            y=dd,
            mode="lines",
            line=dict(color=theme.NEGATIVE, width=2),
            fill="tozeroy",
            fillcolor="rgba(230, 103, 103, 0.18)",
            name="Drawdown",
            hovertemplate="%{y:.1%}",
        )
    )
    fig.update_layout(**theme.base_layout(showlegend=False, margin=dict(l=56, r=24, t=8, b=32)))
    theme.style_axes(fig)
    fig.update_yaxes(tickformat=".0%")
    return fig


# ---------------------------------------------------------------- Comparação


def short(ticker: str) -> str:
    return ticker.removesuffix(".SA")


def dodge(positions: list[float], min_gap: float) -> list[float]:
    """Afasta rótulos que ficariam um em cima do outro, mantendo a ordem vertical.

    Ordena, empurra para cima cada rótulo que ficou perto demais do anterior e
    depois recentra o grupo, para os rótulos não subirem todos.
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    placed = [positions[i] for i in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + min_gap)
    shift = (sum(placed) - sum(positions[i] for i in order)) / len(placed)
    out = [0.0] * len(positions)
    for k, i in enumerate(order):
        out[i] = placed[k] - shift
    return out


NICE_TICKS = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000]


def normalized_figure(norm: pd.DataFrame, colors: dict[str, str], indexes: set[str]) -> go.Figure:
    """Uma linha por ativo, todas começando em 100. Índices tracejados."""
    fig = go.Figure()
    for ticker in norm.columns:
        s = norm[ticker].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s, name=short(ticker), mode="lines",
            line=dict(color=colors[ticker], width=2, dash="dot" if ticker in indexes else "solid"),
            hovertemplate="%{y:.0f}",
        ))

    # Rótulo direto no fim de cada linha. Em eixo log, o Plotly posiciona
    # anotações em log10 do valor, então o cálculo todo é feito em log10.
    log_lo, log_hi = np.log10(norm.min().min()), np.log10(norm.max().max())
    ends = [np.log10(norm[t].dropna().iloc[-1]) for t in norm.columns]
    for ticker, y in zip(norm.columns, dodge(ends, min_gap=0.05 * (log_hi - log_lo))):
        fig.add_annotation(
            x=norm[ticker].dropna().index[-1], y=y, text=short(ticker), showarrow=False,
            xanchor="left", xshift=6, font=dict(size=11, color=theme.INK_2),
        )
    fig.add_hline(y=100, line=dict(color=theme.MUTED, width=1, dash="dot"))
    fig.update_layout(**theme.base_layout(margin=dict(l=56, r=72, t=24, b=32)))
    theme.style_axes(fig)
    fig.update_xaxes(hoverformat="%d/%m/%Y")
    ticks = [v for v in NICE_TICKS if 10**log_lo * 0.9 <= v <= 10**log_hi * 1.1]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[str(v) for v in ticks],
                     title_text="R$ 100 investidos viraram", title_font=dict(size=11, color=theme.MUTED))
    return fig


def risk_return_figure(rr: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """Todos os ativos em cinza como contexto; os selecionados em cor e com nome."""
    fig = go.Figure()
    others = rr[~rr["ticker"].isin(colors)]
    chosen = rr[rr["ticker"].isin(colors)]
    fig.add_trace(go.Scatter(
        x=others["volatility"], y=others["cagr"], mode="markers", name="outros ativos",
        marker=dict(size=9, color=theme.AXIS, line=dict(width=0)),
        text=others["ticker"].map(short), hovertemplate="%{text}<br>vol %{x:.0%} · retorno %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=chosen["volatility"], y=chosen["cagr"], mode="markers+text", name="selecionados",
        marker=dict(size=13, color=[colors[t] for t in chosen["ticker"]], line=dict(width=2, color=theme.SURFACE)),
        text=chosen["ticker"].map(short), textposition="top center", textfont=dict(size=11, color=theme.INK_2),
        hovertemplate="%{text}<br>vol %{x:.0%} · retorno %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(**theme.base_layout(showlegend=False, hovermode="closest"))
    theme.style_axes(fig)
    fig.update_xaxes(tickformat=".0%", title_text="volatilidade anualizada (risco) →",
                     title_font=dict(size=11, color=theme.MUTED))
    fig.update_yaxes(tickformat=".0%", title_text="retorno ao ano →", title_font=dict(size=11, color=theme.MUTED),
                     zeroline=True, zerolinecolor=theme.AXIS)
    return fig


def correlation_figure(corr: pd.DataFrame) -> go.Figure:
    labels = [short(t) for t in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale=theme.DIVERGING,
        text=corr.round(2).values, texttemplate="%{text:.2f}", textfont=dict(color=theme.INK, size=12),
        xgap=2, ygap=2,  # fresta da cor do fundo entre as células
        hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=theme.MUTED), tickvals=[-1, 0, 1]),
    ))
    fig.update_layout(**theme.base_layout(showlegend=False, hovermode="closest", margin=dict(l=72, r=24, t=8, b=40)))
    fig.update_xaxes(showgrid=False, tickfont=dict(color=theme.INK_2))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=theme.INK_2), autorange="reversed")
    return fig


def beta_figure(points: pd.DataFrame, beta: float, alpha: float, color: str, labels: tuple[str, str]) -> go.Figure:
    """Cada ponto é um dia: retorno do índice (x) × retorno da ação (y), mais a reta ajustada."""
    fig = go.Figure(go.Scatter(
        x=points["index"], y=points["asset"], mode="markers", name="dias",
        marker=dict(size=5, color=color, opacity=0.35, line=dict(width=0)),
        customdata=points.index.strftime("%d/%m/%Y"),
        hovertemplate="%{customdata}<br>" + labels[0] + " %{x:.2%} · " + labels[1] + " %{y:.2%}<extra></extra>",
    ))
    x = pd.Series([points["index"].min(), points["index"].max()])
    fig.add_trace(go.Scatter(
        x=x, y=alpha + beta * x, mode="lines", name="reta", line=dict(color=theme.INK, width=2),
        hoverinfo="skip",
    ))
    fig.update_layout(**theme.base_layout(showlegend=False, hovermode="closest"))
    theme.style_axes(fig)
    fig.update_xaxes(tickformat=".0%", title_text=f"retorno diário do {labels[0]}",
                     title_font=dict(size=11, color=theme.MUTED), zeroline=True, zerolinecolor=theme.AXIS)
    fig.update_yaxes(tickformat=".0%", title_text=f"retorno diário de {labels[1]}",
                     title_font=dict(size=11, color=theme.MUTED), zeroline=True, zerolinecolor=theme.AXIS)
    return fig


# ---------------------------------------------------------------- Previsão


def _pct_axes(fig, x_title: str | None = None, y_title: str | None = None, zero: bool = True) -> None:
    title_font = dict(size=11, color=theme.MUTED)
    fig.update_xaxes(tickformat=".0%", title_text=x_title, title_font=title_font,
                     zeroline=zero, zerolinecolor=theme.AXIS)
    fig.update_yaxes(tickformat=".0%", title_text=y_title, title_font=title_font,
                     zeroline=zero, zerolinecolor=theme.AXIS)


def prediction_scatter(bt: pd.DataFrame, color: str, model: str, log: bool = False) -> go.Figure:
    """Previsto (x) × realizado (y). Modelo perfeito = todos os pontos na diagonal.

    Com `log=True` (volatilidade), os dois eixos ficam em escala log: algumas semanas
    de pânico chegam a 300% e, em escala linear, espremeriam todo o resto num canto.
    """
    lo = float(min(bt["y_pred"].min(), bt["y_true"].min()))
    hi = float(max(bt["y_pred"].max(), bt["y_true"].max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="previsão perfeita",
        line=dict(color=theme.MUTED, width=1, dash="dot"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=bt["y_pred"], y=bt["y_true"], mode="markers", name=model,
        marker=dict(size=6, color=color, opacity=0.45, line=dict(width=0)),
        customdata=bt["date"].dt.strftime("%d/%m/%Y"),
        hovertemplate="semana de %{customdata}<br>previsto %{x:.1%} · realizado %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(**theme.base_layout(showlegend=False, hovermode="closest"))
    theme.style_axes(fig)
    _pct_axes(fig, "previsto pelo modelo", "o que aconteceu", zero=not log)
    if log:
        ticks = [v for v in (0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3) if lo * 0.8 <= v <= hi * 1.2]
        fig.update_xaxes(type="log", tickvals=ticks)
        fig.update_yaxes(type="log", tickvals=ticks)
    return fig


def edge_figure(edge: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """Vantagem acumulada sobre a baseline: sobe enquanto o modelo ganha, desce enquanto perde."""
    fig = go.Figure()
    for name in edge.columns:
        fig.add_trace(go.Scatter(
            x=edge.index, y=edge[name], name=name, mode="lines",
            line=dict(color=colors[name], width=2), hovertemplate="%{y:+.2f}",
        ))
    lo, hi = float(edge.min().min()), float(edge.max().max())
    ends = [float(edge[c].iloc[-1]) for c in edge.columns]
    for name, y in zip(edge.columns, dodge(ends, min_gap=0.06 * (hi - lo or 1))):
        fig.add_annotation(x=edge.index[-1], y=y, text=name, showarrow=False, xanchor="left",
                           xshift=6, font=dict(size=11, color=theme.INK_2))
    fig.add_hline(y=0, line=dict(color=theme.MUTED, width=1, dash="dot"))
    fig.update_layout(**theme.base_layout(margin=dict(l=56, r=120, t=24, b=32)))
    theme.style_axes(fig)
    fig.update_xaxes(hoverformat="%d/%m/%Y", range=[edge.index[0], edge.index[-1]])
    fig.update_yaxes(title_text="vantagem acumulada", title_font=dict(size=11, color=theme.MUTED))
    return fig


def importance_figure(importance: pd.Series, color: str) -> go.Figure:
    """Barras horizontais: quanto o LightGBM usou cada feature."""
    imp = importance.sort_values()
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        hovertemplate="%{y}: %{x:.0%}<extra></extra>",
    ))
    fig.update_layout(**theme.base_layout(showlegend=False, hovermode="closest", bargap=0.35,
                                          margin=dict(l=8, r=24, t=8, b=32)))
    theme.style_axes(fig)
    fig.update_xaxes(tickformat=".0%", gridcolor=theme.GRID, showgrid=True)
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(color=theme.INK_2))
    return fig


def overview_figure(ov: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """Um ponto por (ação, modelo). Preenchido = diferença significativa; vazado = pode ser sorte."""
    order = ov.groupby("ticker")["skill"].max().sort_values().index
    labels = [short(t) for t in order]
    fig = go.Figure()
    for name, color in colors.items():
        d = ov[ov["model"] == name].set_index("ticker").loc[order]
        fig.add_trace(go.Scatter(
            x=d["skill"], y=labels, mode="markers", name=name,
            marker=dict(
                size=11, color=[color if s else theme.SURFACE for s in d["significant"]],
                line=dict(color=color, width=2), symbol="circle",
            ),
            customdata=d["p_value"],
            hovertemplate="%{y} · " + name + "<br>skill %{x:+.1%} · p = %{customdata:.3f}<extra></extra>",
        ))
    fig.add_vline(x=0, line=dict(color=theme.MUTED, width=1, dash="dot"))
    fig.update_layout(**theme.base_layout(hovermode="closest", margin=dict(l=8, r=24, t=24, b=40)))
    theme.style_axes(fig)
    fig.update_xaxes(tickformat="+.0%", showgrid=True, gridcolor=theme.GRID,
                     title_text="← pior que a baseline · melhor que a baseline →",
                     title_font=dict(size=11, color=theme.MUTED))
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(color=theme.INK_2))
    return fig
