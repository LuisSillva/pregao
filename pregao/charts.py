"""Monta as figuras Plotly da tela de análise.

Cada função recebe dados prontos e devolve um `go.Figure`. Nenhuma delas sabe
nada de Dash: dá para testar e até usar num notebook.
"""

from __future__ import annotations

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
