"""Ponto de entrada do app.

Rodar localmente:  python -m pregao.app
Em produção:       gunicorn pregao.app:server
"""

import dash
from dash import Dash, html

app = Dash(
    __name__,
    use_pages=True,  # cada arquivo em pregao/pages/ vira uma página
    title="Pregão",
    update_title=None,  # não pisca "Updating..." na aba a cada callback
)
server = app.server  # o Flask por baixo do Dash, que o gunicorn usa

app.layout = html.Div(
    className="shell",
    children=[
        html.Header(
            className="topbar",
            children=[
                html.Div("Pregão", className="brand"),
                html.Nav(
                    className="nav",
                    children=[
                        dash.dcc.Link(page["name"], href=page["relative_path"], className="nav-link")
                        for page in dash.page_registry.values()
                    ],
                ),
            ],
        ),
        html.Main(dash.page_container, className="content"),
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
