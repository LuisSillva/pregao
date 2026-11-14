"""Cores e estilo dos gráficos, num lugar só.

A paleta das séries foi validada para daltonismo (as três primeiras cores
continuam distinguíveis entre si nos três tipos mais comuns). Por isso as velas
do gráfico são neutras: alta = vazada, queda = preenchida. Assim as cores ficam
livres para as médias móveis, e alta/queda não depende de distinguir verde de
vermelho, que é justamente o par que daltônicos mais confundem.
"""

SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

SERIES = ["#3987e5", "#d95926", "#199e70"]  # azul, laranja, verde-água
NEGATIVE = "#e66767"  # polo negativo (quedas, drawdown)
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def base_layout(**overrides) -> dict:
    layout = dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=12),
        margin=dict(l=56, r=72, t=24, b=32),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PAGE, bordercolor=AXIS, font=dict(color=INK, family=FONT)),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(color=INK_2)),
    )
    layout.update(overrides)
    return layout


def style_axes(fig) -> None:
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
