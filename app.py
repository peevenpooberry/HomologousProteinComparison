#!/usr/bin/python3

import os
import base64
import json
import subprocess
import tempfile
import numpy as np
import zipfile
import io as io_module
import sqlite3
import threading
import time

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update, ALL
import dash_bootstrap_components as dbc
import dash_bio as dashbio
from dash_bio.utils import PdbParser as DashPdbParser, create_mol3d_style
import plotly.graph_objects as go
from Bio.PDB import MMCIFParser, PDBIO, Select

# ── Constants ─────────────────────────────────────────────────────────────────
MUSCLE_PATH  = "/code/MUSCLE/muscle-linux-x86.v5.3"
P2RANK_PATH  = "/code/p2rank_2.5.1"
SESSION_BASE = "/tmp/sessions"
DB_PATH      = "/tmp/sessions.db"

# ── SQLite helpers ────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            key TEXT PRIMARY KEY,
            value TEXT,
            expires_at REAL
        )
    """)
    conn.commit()
    return conn

def db_set(key: str, value: str, ttl_seconds: int = 86400):
    expires = time.time() + ttl_seconds
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (key, value, expires_at) VALUES (?, ?, ?)",
            (key, value, expires)
        )

def db_get(key: str) -> str | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM sessions WHERE key=? AND expires_at > ?",
            (key, time.time())
        ).fetchone()
    return row[0] if row else None

def db_hset(key: str, mapping: dict):
    db_set(key, json.dumps(mapping))

def db_hgetall(key: str) -> dict:
    val = db_get(key)
    return json.loads(val) if val else {}

def redis_session_key(session_name, suffix):
    return f"session:{session_name}:{suffix}"

def store_file_in_db(session_name, slot, filename):
    db_hset(f"session:{session_name}:file{slot}", {
        "filename": filename,
        "path": f"/tmp/sessions/{session_name}/Input/{filename}"
    })

def write_file_to_disk(session_dir, filename, contents_b64):
    _, content_string = contents_b64.split(",")
    decoded = base64.b64decode(content_string)
    path = os.path.join(session_dir, filename)
    with open(path, "wb") as f:
        f.write(decoded)
    return path

def list_completed_sessions() -> list[dict]:
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT key, value FROM sessions WHERE key LIKE 'session:%:meta'"
            ).fetchall()
        sessions = []
        for key, value in rows:
            try:
                meta = json.loads(value)
                if meta.get("status") == "complete":
                    name = key.split(":")[1]
                    sessions.append({"label": name, "value": name})
            except Exception:
                continue
        return sessions
    except Exception:
        return []

# ── Gaussian helpers ──────────────────────────────────────────────────────────
def gaussian_curve(mean, std, x_range=(0, 100), n=300):
    x = np.linspace(*x_range, n)
    y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    return x, y

def make_gaussian_fig(mean, std):
    x, y   = gaussian_curve(mean, std)
    y_norm = y / y.max() if y.max() > 0 else y
    mask   = (x >= mean - std) & (x <= mean + std)
    shade_x, shade_y = x[mask], y_norm[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([[shade_x[0]], shade_x, [shade_x[-1]]]),
        y=np.concatenate([[0], shade_y, [0]]),
        fill="toself", fillcolor="rgba(55,138,221,0.15)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_norm, line=dict(color="#378ADD", width=2),
        showlegend=False,
        hovertemplate="pLDDT: %{x:.1f}<br>Weight: %{y:.3f}<extra></extra>"
    ))
    fig.add_vline(x=mean, line=dict(color="#185FA5", width=1.5, dash="dot"))
    fig.add_vline(x=mean - std, line=dict(color="#85B7EB", width=1, dash="dash"))
    fig.add_vline(x=mean + std, line=dict(color="#85B7EB", width=1, dash="dash"))
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8), height=180,
        xaxis=dict(title="pLDDT score", range=[0, 100], showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(title="weight", showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                   tickfont=dict(size=11), range=[0, 1.1]),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Georgia, serif", size=11),
    )
    return fig

def build_residue_plot(session_name: str, meta: dict, slot: int = 0):
    """Build a toggleable per-residue score plot from session meta + scores."""
    try:
        seq_conservation   = json.loads(meta.get("sequence_conservation", "[]"))
        plddt_conservation = json.loads(meta.get("plddt_conservation",    "[]"))
        p2rank_conservation= json.loads(meta.get("p2rank_conservation",   "[]"))

        if not seq_conservation:
            return html.Div("No conservation data found.",
                            style={"color": "#ccc", "fontSize": "13px",
                                   "textAlign": "center", "paddingTop": "120px"})

        scores_json  = db_get(redis_session_key(session_name, f"scores:slot{slot}"))
        final_scores = json.loads(scores_json) if scores_json else {}
        final_scores_list = [
            float(final_scores.get(str(i), 0))
            for i in range(len(seq_conservation))
        ]

        x = list(range(1, len(seq_conservation) + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=seq_conservation,
            name="Sequence conservation",
            mode="lines",
            line=dict(color="#2196F3", width=1.5),
            hovertemplate="Residue %{x}<br>Seq: %{y:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=x, y=plddt_conservation,
            name="pLDDT conservation",
            mode="lines",
            line=dict(color="#4CAF50", width=1.5),
            hovertemplate="Residue %{x}<br>pLDDT: %{y:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=x, y=p2rank_conservation,
            name="P2Rank conservation",
            mode="lines",
            line=dict(color="#FF9800", width=1.5),
            hovertemplate="Residue %{x}<br>P2Rank: %{y:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=x, y=final_scores_list,
            name="Final score",
            mode="lines",
            line=dict(color="#E91E63", width=2.5),
            hovertemplate="Residue %{x}<br>Score: %{y:.3f}<extra></extra>"
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=8, r=8, t=8, b=8),
            xaxis=dict(title="Residue index", showgrid=False, tickfont=dict(size=11)),
            yaxis=dict(title="Score", showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                       tickfont=dict(size=11), range=[0, 1.05]),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Georgia, serif", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=11)),
            hovermode="x unified",
        )

        return dcc.Graph(
            figure=fig,
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]
            },
        )

    except Exception as e:
        import traceback
        print(f"[PLOT ERROR] {traceback.format_exc()}")
        return dbc.Alert(f"Error building plot: {e}", color="danger",
                         style={"fontSize": "12px", "margin": 0})


# ── Slider config ─────────────────────────────────────────────────────────────
SLIDERS = [
    ("seq-weight",    "Sequence Conservation weight (a)",       0.0, 2.0, 0.01, 0.3),
    ("plddt-weight",  "pLDDT Score Conservation weight (b)",    0.0, 2.0, 0.01, 0.2),
    ("p2rank-weight", "P2Rank Score Conservation weight (c)",   0.0, 2.0, 0.01, 0.5),
    ("gauss-mean",    "Gaussian mean (pLDDT)",                  0,   100, 1,    70),
    ("gauss-std",     "Gaussian std dev (pLDDT)",               1,   30,  0.5,  5),
]

def make_slider_row(slider_id, label, min_val, max_val, step, default):
    return html.Div(style={"marginBottom": "1.1rem"}, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "3px"}, children=[
            html.Label(label, style={"fontSize": "12px", "color": "#6c757d"}),
            html.Span(id=f"val-{slider_id}", children=str(default),
                      style={"fontSize": "12px", "fontWeight": "500",
                             "color": "#212529", "fontFamily": "monospace"}),
        ]),
        dcc.Slider(id=f"slider-{slider_id}", min=min_val, max=max_val,
                   step=step, value=default, marks=None,
                   tooltip={"always_visible": False}),
    ])

# ── Initialize app ────────────────────────────────────────────────────────────
external_stylesheets = [
    dbc.themes.CERULEAN,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"
]
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    prevent_initial_callbacks="initial_duplicate"
)
server = app.server

# ── Layout ────────────────────────────────────────────────────────────────────
CARD = {"border": "0.5px solid #dee2e6", "borderRadius": "12px", "backgroundColor": "white"}

app.layout = dbc.Container(fluid=True, style={"backgroundColor": "#f5f5f3", "minHeight": "100vh", "padding": "1.5rem"}, children=[
    html.Div(
    style={
        "marginBottom": "1.5rem",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center"
    },
    children=[

        # Left: Title block
        html.Div([
            html.H1("Homology Guided Protein Binding Site Analysis", style={
                "fontFamily": "'Georgia', serif",
                "fontWeight": "400",
                "fontSize": "26px",
                "marginBottom": "2px",
                "letterSpacing": "-0.3px"
            }),
            html.P("Structure & Sequence Conservation-based scoring pipeline",
                   style={"color": "#888", "fontSize": "13px", "margin": 0}),
        ]),

        # Right: GitHub link
        html.A(
            href="https://github.com/peevenpooberry/HomologousProteinComparison#",
            target="_blank",
            children=html.I(className="bi bi-github"),
            style={
                "color": "#333",
                "fontSize": "36px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "padding": "6px",
                "borderRadius": "8px",
                "transition": "transform 0.15s ease"
                }
            )
    ]),

        # Row 1: Submission + Parameters + Gaussian
        dbc.Row(className="g-3 mb-3", children=[
        dbc.Col(width=12, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("Past sessions", style={
                    "fontWeight": "500", "marginBottom": "0.75rem",
                    "fontSize": "13px", "textTransform": "uppercase",
                    "letterSpacing": "0.05em", "color": "#888"
                }),
                dbc.Row(className="g-2", children=[
                    dbc.Col(children=[
                        dcc.Dropdown(
                            id="past-session-dropdown",
                            placeholder="Select a past session...",
                            style={"fontSize": "13px"}
                        ),
                    ]),
                    dbc.Col(width="auto", children=[
                        dbc.Button("Load", id="load-session-btn",
                                color="primary", outline=True, size="sm",
                                style={"borderRadius": "8px", "fontSize": "12px"}),
                    ]),
                    dbc.Col(width="auto", children=[
                        dbc.Button("↻", id="refresh-sessions-btn",
                                color="secondary", outline=True, size="sm",
                                style={"borderRadius": "8px", "fontSize": "12px"}),
                    ]),
                ]),
                html.Div(id="load-session-status",
                        style={"marginTop": "0.5rem", "minHeight": "24px"}),
            ])])
        ]),
    ]),
    dbc.Row(className="g-3 mb-3", children=[
        dbc.Col(width=4, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("Session", style={"fontWeight": "500", "marginBottom": "1rem", "fontSize": "13px",
                                          "textTransform": "uppercase", "letterSpacing": "0.05em", "color": "#888"}),
                html.Label("Session name", style={"fontSize": "12px", "color": "#6c757d", "marginBottom": "3px"}),
                dbc.Input(id="session-name", placeholder="e.g. Trypsin_v1", type="text",
                          style={"marginBottom": "1.1rem", "borderRadius": "8px", "fontSize": "13px"}),
                html.Hr(style={"borderColor": "#eee", "margin": "0.75rem 0"}),
                html.H6("Structure files", style={"fontWeight": "500", "marginBottom": "0.75rem", "fontSize": "13px",
                                                   "textTransform": "uppercase", "letterSpacing": "0.05em", "color": "#888"}),
                html.P("Upload at least 2 .pdb or .cif files to enable analysis.",
                       style={"fontSize": "11px", "color": "#aaa", "marginBottom": "0.75rem"}),
                html.Label("Structure 1", style={"fontSize": "12px", "color": "#6c757d", "marginBottom": "3px"}),
                dcc.Upload(id="upload-file1", accept=".cif,.pdb", multiple=False,
                    children=html.Div(id="upload-file1-label", children=[
                        html.Span("↑ ", style={"color": "#ccc"}),
                        html.Span("Click or drag .pdb/.cif", style={"fontSize": "12px", "color": "#999"}),
                    ]),
                    style={"border": "1.5px dashed #ddd", "borderRadius": "8px",
                           "padding": "0.6rem 1rem", "cursor": "pointer", "marginBottom": "0.75rem"}),
                html.Label("Structure 2", style={"fontSize": "12px", "color": "#6c757d", "marginBottom": "3px"}),
                dcc.Upload(id="upload-file2", accept=".cif,.pdb", multiple=False,
                    children=html.Div(id="upload-file2-label", children=[
                        html.Span("↑ ", style={"color": "#ccc"}),
                        html.Span("Click or drag .pdb/.cif", style={"fontSize": "12px", "color": "#999"}),
                    ]),
                    style={"border": "1.5px dashed #ddd", "borderRadius": "8px",
                           "padding": "0.6rem 1rem", "cursor": "pointer", "marginBottom": "0.75rem"}),
                html.Label("Additional structures (optional)", style={"fontSize": "12px", "color": "#6c757d", "marginBottom": "3px"}),
                dcc.Upload(id="upload-extra", accept=".cif,.pdb", multiple=True,
                    children=html.Div([
                        html.Span("↑ ", style={"color": "#ccc"}),
                        html.Span("Click or drag multiple files", style={"fontSize": "12px", "color": "#999"}),
                    ]),
                    style={"border": "1.5px dashed #ddd", "borderRadius": "8px",
                           "padding": "0.6rem 1rem", "cursor": "pointer", "marginBottom": "1rem"}),
                html.Div(id="file-status", style={"marginBottom": "0.75rem", "minHeight": "24px"}),
                html.Hr(style={"borderColor": "#eee", "margin": "0.75rem 0"}),
                dbc.Button("Run analysis", id="run-btn", color="primary", size="sm", disabled=True,
                           style={"width": "100%", "borderRadius": "8px", "fontWeight": "500", "fontSize": "13px"}),
                html.Div(id="run-status", style={"marginTop": "0.5rem", "minHeight": "24px"}),
            ])])
        ]),
            dbc.Col(width=5, children=[
                    dbc.Card(style=CARD, children=[
                        dbc.CardBody(
                            style={"padding": "1.25rem"},
                            children=[
                                html.H6(
                                    "Scoring weights",
                                    style={
                                        "fontWeight": "500",
                                        "marginBottom": "1rem",
                                        "fontSize": "13px",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.05em",
                                        "color": "#888",
                                    },
                                ),

                                html.H6("Final Score", style={"marginBottom": "0.5rem"}),

                                dcc.Markdown(
                                    r"""
                $$
                \log(S) = a \cdot \log(\text{Sequence Conservation})
                + b \cdot \log(\text{PLDDT Score Conservation})
                + c \cdot \log(\text{P2Rank Score Conservation})
                $$
                """,
                                    mathjax=True,
                                    style={"fontSize": "16px"},
                                ),

                                *[make_slider_row(*s) for s in SLIDERS],
                            ],
                        )
                    ])
                ]),
        dbc.Col(width=3, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("pLDDT gaussian weight", style={"fontWeight": "500", "marginBottom": "0.5rem",
                                                         "fontSize": "13px", "textTransform": "uppercase",
                                                         "letterSpacing": "0.05em", "color": "#888"}),
                html.P("Residues with pLDDT near the mean receive higher weight. Shaded band shows ±1 std dev.",
                       style={"fontSize": "11px", "color": "#999", "marginBottom": "0.75rem", "lineHeight": "1.5"}),
                dcc.Graph(id="gaussian-plot", figure=make_gaussian_fig(70, 5),
                          config={"displayModeBar": False}, style={"margin": "0 -8px"}),
                html.Div(id="gaussian-stats", style={"marginTop": "0.5rem"}),
            ])])
        ]),
    ]),
        html.Div(
        style={
            "margin": "1.5rem 0 1.0rem 0",
            "borderTop": "1px solid #e5e7eb",
            "position": "relative"
        },
        children=[
            html.Span(
                "Visualization",
                style={
                    "position": "absolute",
                    "top": "-10px",
                    "left": "50%",
                    "transform": "translateX(-50%)",
                    "background": "#f5f5f3",
                    "padding": "0 10px",
                    "fontSize": "11px",
                    "color": "#999",
                    "letterSpacing": "0.05em",
                    "textTransform": "uppercase"
                }
            )
        ]),

    # Row 2: Mol3D + Alignment
    dbc.Row(className="g-3 mb-3", children=[
        dbc.Col(width=6, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("Structure viewer", style={"fontWeight": "500", "marginBottom": "0.75rem",
                                                    "fontSize": "13px", "textTransform": "uppercase",
                                                    "letterSpacing": "0.05em", "color": "#888"}),
                html.Div(id="mol3d-selector", style={"marginBottom": "0.5rem"}),
                html.Div(id="mol3d-container", style={"minHeight": "420px"}, children=[
                    html.Div(style={"height": "420px", "display": "flex", "alignItems": "center",
                                    "justifyContent": "center", "color": "#ccc", "fontSize": "13px",
                                    "border": "1px dashed #eee", "borderRadius": "8px"},
                             children="Upload structures to view")
                ]),
            ])])
        ]),
        dbc.Col(width=6, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("Sequence alignment", style={"fontWeight": "500", "marginBottom": "0.75rem",
                                                      "fontSize": "13px", "textTransform": "uppercase",
                                                      "letterSpacing": "0.05em", "color": "#888"}),
                html.Div(id="alignment-container", style={"minHeight": "420px"}, children=[
                    html.Div(style={"height": "420px", "display": "flex", "alignItems": "center",
                                    "justifyContent": "center", "color": "#ccc", "fontSize": "13px",
                                    "border": "1px dashed #eee", "borderRadius": "8px"},
                             children="Alignment will appear after analysis runs")
                ]),
            ])])
        ]),
    ]),

    # Row 3: Per-residue plot
    dbc.Row(className="g-3 mb-3", children=[
        dbc.Col(width=12, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("Per-aligned residue scores", style={"fontWeight": "500", "marginBottom": "0.75rem",
                                                      "fontSize": "13px", "textTransform": "uppercase",
                                                      "letterSpacing": "0.05em", "color": "#888"}),
                html.Div(id="residue-plot-container", style={"minHeight": "300px"}, children=[
                    html.Div(style={"height": "300px", "display": "flex", "alignItems": "center",
                                    "justifyContent": "center", "color": "#ccc", "fontSize": "13px",
                                    "border": "1px dashed #eee", "borderRadius": "8px"},
                             children="Run analysis to see per-residue scores")
                ]),
            ])])
        ]),
    ]),

    # Row 4: Past sessions + Downloads
    dbc.Row(className="g-3 mb-3", children=[
        dbc.Col(width=6, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("Download results", style={"fontWeight": "500", "marginBottom": "0.75rem",
                                                    "fontSize": "13px", "textTransform": "uppercase",
                                                    "letterSpacing": "0.05em", "color": "#888"}),
                html.P("Downloads a ZIP containing the summary JSON and MSA FASTA.",
                       style={"fontSize": "11px", "color": "#aaa", "marginBottom": "0.75rem"}),
                dbc.Button("Download ZIP", id="dl-zip-btn", color="primary", outline=True, size="sm",
                           style={"borderRadius": "8px", "fontSize": "12px"}),
                dcc.Download(id="download-zip"),
                html.Div(id="download-status", style={"marginTop": "0.5rem", "minHeight": "24px"}),
            ])])
        ]),
    ]),

    # Polling + Stores
    dcc.Interval(id="poll-interval", interval=3000, n_intervals=0, disabled=True),
    dcc.Store(id="file1-store"),
    dcc.Store(id="file2-store"),
    dcc.Store(id="extra-store"),
    dcc.Store(id="session-store"),
    dcc.Store(id="mol3d-active", data=1),
])
# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(Output("val-seq-weight",    "children"), Input("slider-seq-weight",    "value"))
def _v0(v): return f"{v:.2f}"
@app.callback(Output("val-plddt-weight",  "children"), Input("slider-plddt-weight",  "value"))
def _v1(v): return f"{v:.2f}"
@app.callback(Output("val-p2rank-weight", "children"), Input("slider-p2rank-weight", "value"))
def _v2(v): return f"{v:.2f}"
@app.callback(Output("val-gauss-mean",    "children"), Input("slider-gauss-mean",    "value"))
def _v3(v): return f"{v:.0f}"
@app.callback(Output("val-gauss-std",     "children"), Input("slider-gauss-std",     "value"))
def _v4(v): return f"{v:.1f}"

@app.callback(
    Output("gaussian-plot",  "figure"),
    Output("gaussian-stats", "children"),
    Input("slider-gauss-mean", "value"),
    Input("slider-gauss-std",  "value"),
)
def update_gaussian_cb(mean, std):
    fig = make_gaussian_fig(mean, std)
    stats = html.Div(style={"display": "flex", "gap": "12px"}, children=[
        html.Div(style={"flex": "1", "background": "#f8f9fa", "borderRadius": "6px",
                        "padding": "6px 10px", "textAlign": "center"}, children=[
            html.P("mean", style={"fontSize": "10px", "color": "#999", "margin": "0"}),
            html.P(f"{mean:.0f}", style={"fontSize": "16px", "fontWeight": "500",
                                          "margin": "0", "fontFamily": "monospace"}),
        ]),
        html.Div(style={"flex": "1", "background": "#f8f9fa", "borderRadius": "6px",
                        "padding": "6px 10px", "textAlign": "center"}, children=[
            html.P("±1σ range", style={"fontSize": "10px", "color": "#999", "margin": "0"}),
            html.P(f"{max(0, mean-std):.0f}–{min(100, mean+std):.0f}",
                   style={"fontSize": "16px", "fontWeight": "500",
                          "margin": "0", "fontFamily": "monospace"}),
        ]),
    ])
    return fig, stats

@app.callback(
    Output("file1-store", "data"), Output("upload-file1-label", "children"),
    Input("upload-file1", "contents"), State("upload-file1", "filename"),
    prevent_initial_call=True,
)
def store_file1(contents, filename):
    if not contents: return no_update, no_update
    if os.path.splitext(filename)[1].lower() not in [".pdb", ".cif"]:
        return None, dbc.Badge("Invalid file type", color="danger", style={"fontSize": "11px"})
    return {"filename": filename, "contents": contents}, [
        html.Span("✓ ", style={"color": "#28a745", "fontSize": "12px"}),
        html.Span(filename, style={"fontSize": "12px", "color": "#495057"}),
    ]

@app.callback(
    Output("file2-store", "data"), Output("upload-file2-label", "children"),
    Input("upload-file2", "contents"), State("upload-file2", "filename"),
    prevent_initial_call=True,
)
def store_file2(contents, filename):
    if not contents: return no_update, no_update
    if os.path.splitext(filename)[1].lower() not in [".pdb", ".cif"]:
        return None, dbc.Badge("Invalid file type", color="danger", style={"fontSize": "11px"})
    return {"filename": filename, "contents": contents}, [
        html.Span("✓ ", style={"color": "#28a745", "fontSize": "12px"}),
        html.Span(filename, style={"fontSize": "12px", "color": "#495057"}),
    ]

@app.callback(
    Output("extra-store", "data"),
    Input("upload-extra", "contents"), State("upload-extra", "filename"),
    prevent_initial_call=True,
)
def store_extra(contents_list, filenames):
    if not contents_list: return no_update
    valid = [{"filename": fn, "contents": c} for fn, c in zip(filenames, contents_list)
             if os.path.splitext(fn)[1].lower() in [".pdb", ".cif"]]
    return valid or None

@app.callback(
    Output("run-btn", "disabled"), Output("file-status", "children"),
    Input("session-name", "value"), Input("file1-store", "data"), Input("file2-store", "data"),
)
def toggle_run_button(session_name, file1, file2):
    has_session = bool(session_name and session_name.strip())
    has_files   = bool(file1 and file2)
    if not has_session and not has_files:
        return True, None
    if not has_session:
        return True, dbc.Alert("Enter a session name to continue.", color="warning",
                               style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})
    if not has_files:
        return True, dbc.Alert("Upload at least 2 structure files.", color="warning",
                               style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})
    return False, dbc.Alert("Ready to run.", color="success",
                            style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})

@app.callback(
    Output("run-status", "children"), Output("session-store", "data"),
    Output("poll-interval", "disabled"),
    Input("run-btn", "n_clicks"),
    State("session-name", "value"), State("file1-store", "data"),
    State("file2-store", "data"), State("extra-store", "data"),
    State("slider-seq-weight", "value"), State("slider-plddt-weight", "value"),
    State("slider-p2rank-weight", "value"), State("slider-gauss-mean", "value"),
    State("slider-gauss-std", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, session_name, file1, file2, extra,
                 w_seq, w_plddt, w_p2rank, g_mean, g_std):
    session_name = session_name.strip()
    session_dir  = os.path.join(SESSION_BASE, session_name)
    input_dir    = os.path.join(session_dir, "Input")
    work_dir     = os.path.join(session_dir, "Work")
    output_dir   = os.path.join(session_dir, "Output")
    for d in [input_dir, work_dir, output_dir]:
        os.makedirs(d, exist_ok=True)

    all_files = [file1, file2] + (extra or [])
    for i, f in enumerate(all_files, start=1):
        write_file_to_disk(input_dir, f["filename"], f["contents"])
        store_file_in_db(session_name, i, f["filename"])

    db_hset(redis_session_key(session_name, "meta"), {
        "status": "running", "session_name": session_name,
        "output_dir": output_dir, "file_count": len(all_files),
    })

    cmd = [
        "python3", "/code/main_workflow/calc_stats.py",
        "-n", session_name, "-m", MUSCLE_PATH, "-p", P2RANK_PATH,
        "-i", input_dir, "-w", work_dir, "-o", output_dir,
        "-a", str(w_seq), "-b", str(w_plddt), "-c", str(w_p2rank),
        "-u", str(g_mean), "-s", str(g_std),
    ]

    def load_session_results(sname, odir):
        summary_path = os.path.join(odir, f"{sname}_summary.json")
        if not os.path.exists(summary_path):
            return None
        with open(summary_path) as f:
            root = json.load(f)
        result = root["result"]
        for idx, protein in enumerate(result["protein_files"]):
            scores    = protein["final_score_per_residue"]
            score_key = redis_session_key(sname, f"scores:slot{idx}")
            db_set(score_key, json.dumps({str(i): float(s) for i, s in enumerate(scores)}))
            print(f"[SCORES] slot{idx} → {protein['file_name']} ({len(scores)} residues)")
        db_hset(redis_session_key(sname, "meta"), {
            "status": "complete", "output_dir": odir,
            "final_score":            json.dumps(result["final_score"]),
            "sequence_conservation":  json.dumps(result["sequence_conservation"]),
            "plddt_conservation":     json.dumps(result["PLDDT_score_conservation"]),
            "p2rank_conservation":    json.dumps(result["P2Rank_score_conservation"]),
        })
        return result

    def run_and_update():
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            print("[CMD]", " ".join(cmd))
            print("[STDOUT]", res.stdout)
            print("[STDERR]", res.stderr)
            if res.returncode == 0:
                parsed = load_session_results(session_name, output_dir)
                if parsed is None:
                    db_hset(redis_session_key(session_name, "meta"),
                            {"status": "error", "stderr": "Output JSON not found."})
            else:
                db_hset(redis_session_key(session_name, "meta"),
                        {"status": "error", "stderr": res.stderr[-2000:]})
        except Exception as e:
            db_hset(redis_session_key(session_name, "meta"),
                    {"status": "error", "stderr": str(e)})

    threading.Thread(target=run_and_update, daemon=True).start()

    msg = dbc.Alert([
        html.Span(dbc.Spinner(size="sm", color="primary"), style={"marginRight": "8px"}),
        f"Session '{session_name}' running…"
    ], color="info", style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})
    return msg, session_name, False

@app.callback(
    Output("run-status",             "children",  allow_duplicate=True),
    Output("poll-interval",          "disabled",  allow_duplicate=True),
    Output("alignment-container",    "children",  allow_duplicate=True),
    Output("mol3d-container",        "children",  allow_duplicate=True),
    Output("residue-plot-container", "children",  allow_duplicate=True),
    Input("poll-interval",           "n_intervals"),
    State("session-store",  "data"), State("file1-store", "data"),
    State("file2-store",    "data"), State("extra-store",  "data"),
    State("mol3d-active",   "data"),
    prevent_initial_call=True,
)
def poll_results(n_intervals, session_name, file1, file2, extra, active_slot):
    if not session_name:
        return no_update, True, no_update, no_update, no_update
    meta   = db_hgetall(redis_session_key(session_name, "meta"))
    status = meta.get("status", "unknown")
    if status == "running":
        return no_update, False, no_update, no_update, no_update
    if status == "error":
        err = meta.get("stderr", "Unknown error")
        return dbc.Alert(f"Analysis failed: {err[:300]}", color="danger",
                         style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0}), \
               True, no_update, no_update, no_update
    if status == "complete":
        output_dir = meta.get("output_dir", "")
        status_msg = dbc.Alert("Analysis complete.", color="success",
                               style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})
        msa_path = os.path.join(output_dir, f"{session_name}_msa.fasta")
        alignment_component = no_update
        if os.path.exists(msa_path):
            with open(msa_path) as f:
                aln_data = f.read()
            alignment_component = dashbio.AlignmentChart(
                id="alignment-chart", data=aln_data, height=420,
                tilewidth=20, showconservation=True, showgap=True,
            )
        slot      = (active_slot or 1) - 1
        all_files = [f for f in [file1, file2] if f] + (extra or [])
        mol3d_component = no_update
        if all_files:
            idx             = min(slot, len(all_files) - 1)
            mol3d_component = render_mol3d(all_files[idx], session_name, slot=idx)
        residue_plot = build_residue_plot(session_name, meta, slot=slot)
        return status_msg, True, alignment_component, mol3d_component, residue_plot
    return no_update, False, no_update, no_update, no_update

class FirstChainSelect(Select):
    """Only write the first protein chain, skip heteroatoms and water."""
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        # Skip HETATM records (ligands, water)
        return residue.id[0] == " "

def cif_bytes_to_pdb_bytes(cif_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="wb") as cif_tmp:
        cif_tmp.write(cif_bytes)
        cif_tmp_path = cif_tmp.name
    try:
        parser    = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", cif_tmp_path)

        # Get the first chain ID from the first model
        first_model = next(iter(structure))
        chains      = list(first_model.get_chains())

        if not chains:
            raise ValueError("No chains found in structure")

        # Find first protein chain (single character ID, has standard residues)
        protein_chain = None
        for chain in chains:
            if len(chain.id) == 1:
                residues = list(chain.get_residues())
                # Check it has standard amino acid residues
                std_res = [r for r in residues if r.id[0] == " "]
                if std_res:
                    protein_chain = chain
                    break

        if protein_chain is None:
            # Fallback — just take first single-char chain
            protein_chain = next((c for c in chains if len(c.id) == 1), chains[0])

        pdbio = PDBIO()
        pdbio.set_structure(structure)
        buf = io_module.StringIO()
        pdbio.save(buf, FirstChainSelect(protein_chain.id))
        return buf.getvalue().encode()

    finally:
        os.unlink(cif_tmp_path)

def render_mol3d(file_data: dict, session_name: str | None = None, slot: int = 0):
    if not file_data:
        return html.Div("No structure loaded.", style={
            "height": "420px", "display": "flex", "alignItems": "center",
            "justifyContent": "center", "color": "#ccc", "fontSize": "13px"
        })

    tmp_path = None
    try:
        _, content_string = file_data["contents"].split(",")
        decoded = base64.b64decode(content_string)
        suffix  = os.path.splitext(file_data["filename"])[1].lower()

        # Convert CIF → PDB in memory before passing to DashPdbParser
        if suffix == ".cif":
            pdb_bytes = cif_bytes_to_pdb_bytes(decoded)
        elif suffix == ".pdb":
            pdb_bytes = decoded
        else:
            return dbc.Alert(f"Unsupported format: {suffix}", color="warning",
                             style={"fontSize": "12px", "margin": 0})

        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="wb")
        tmp.write(pdb_bytes)
        tmp.flush()
        tmp.close()
        tmp_path = tmp.name

        parser   = DashPdbParser(tmp_path)
        mol_data = parser.mol3d_data()

        if not mol_data or not mol_data.get("atoms"):
            return dbc.Alert("Parser returned no atoms — is the structure file valid?",
                             color="warning", style={"fontSize": "12px", "margin": 0})

        styles = create_mol3d_style(
            mol_data["atoms"],
            visualization_type="cartoon",
            color_element="residue_type"
        )

        if session_name:
            score_key = redis_session_key(session_name, f"scores:slot{slot}")
            scores_json = db_get(score_key)

            if scores_json:
                scores = json.loads(scores_json)
                
                # Build residue order
                residue_order = []
                last_res = None
                for atom in mol_data["atoms"]:
                    res_id = (atom.get("chain"), atom.get("residue_index"))
                    if res_id != last_res:
                        residue_order.append(res_id)
                        last_res = res_id

                res_to_score = {
                    res_id: float(scores.get(str(i), 0.5))
                    for i, res_id in enumerate(residue_order)
                }

                # Build styles dict keyed by atom serial with score-based color
                styles_dict = {}
                for atom in mol_data["atoms"]:
                    res_id = (atom.get("chain"), atom.get("residue_index"))
                    score  = res_to_score.get(res_id, 0.5)

                    if score >= 0.5:
                        t     = (score - 0.5) * 2
                        r_val, g_val, b_val = 255, int(255 * (1 - t)), int(255 * (1 - t))
                    else:
                        t     = score * 2
                        r_val, g_val, b_val = int(255 * t), int(255 * t), 255

                    styles_dict[atom["serial"]] = {
                        "visualization_type": "cartoon",
                        "color": f"#{r_val:02x}{g_val:02x}{b_val:02x}"
                    }

            else:
                # No scores yet — default residue type coloring
                styles_dict = {
                    atom["serial"]: {"visualization_type": "cartoon", "color": "#6baed6"}
                    for atom in mol_data["atoms"]
                }

            return dashbio.Molecule3dViewer(
                id="mol3d-viewer",
                modelData=mol_data,
                styles=styles_dict,
                height=420,
                backgroundColor="#fafafa",
                selectionType="atom",
                )

    except Exception as e:
        import traceback
        return dbc.Alert(f"Error loading structure: {e}", color="danger",
                         style={"fontSize": "12px", "margin": 0})
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    return dbc.Alert("Unknown rendering state.", color="warning",
                 style={"fontSize": "12px", "margin": 0})

@app.callback(
    Output("mol3d-container", "children"),
    Output("mol3d-selector",  "children"),
    Input("file1-store",  "data"), Input("file2-store", "data"),
    Input("extra-store",  "data"), Input("mol3d-active", "data"),
    State("session-store", "data"),
    prevent_initial_call=True,
)
def update_mol3d(file1, file2, extra, active_slot, session_name):
    all_files = [f for f in [file1, file2] if f] + (extra or [])
    if not all_files:
        return html.Div("Upload structures to view.",
                        style={"height": "420px", "display": "flex", "alignItems": "center",
                               "justifyContent": "center", "color": "#ccc", "fontSize": "13px",
                               "border": "1px dashed #eee", "borderRadius": "8px"}), []
    selector = html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}, children=[
        dbc.Button(f["filename"], id={"type": "mol3d-btn", "index": i + 1},
                   size="sm", outline=(i + 1 != active_slot), color="primary",
                   style={"fontSize": "11px", "borderRadius": "6px", "padding": "2px 10px"})
        for i, f in enumerate(all_files)
    ])
    idx    = min((active_slot or 1) - 1, len(all_files) - 1)
    viewer = render_mol3d(all_files[idx], session_name, slot=idx)
    return viewer, selector

@app.callback(
    Output("mol3d-active", "data"),
    Input({"type": "mol3d-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def switch_mol3d(n_clicks):
    if not any(n_clicks): return no_update
    return ctx.triggered_id["index"]

@app.callback(
    Output("past-session-dropdown", "options"),
    Input("refresh-sessions-btn",   "n_clicks"),
    Input("session-store",          "data"),
)
def refresh_sessions(n_clicks, session_store):
    return list_completed_sessions()

@app.callback(
    Output("session-store",          "data",     allow_duplicate=True),
    Output("load-session-status",    "children"),
    Output("alignment-container",    "children", allow_duplicate=True),
    Output("residue-plot-container", "children", allow_duplicate=True),
    Input("load-session-btn",        "n_clicks"),
    State("past-session-dropdown",   "value"),
    prevent_initial_call=True,
)
def load_past_session(n_clicks, session_name):
    if not session_name:
        return no_update, dbc.Alert("Select a session first.", color="warning",
                                    style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0}), \
               no_update, no_update
    meta = db_hgetall(redis_session_key(session_name, "meta"))
    if not meta:
        return no_update, dbc.Alert("Session not found.", color="danger",
                                    style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0}), \
               no_update, no_update
    output_dir = meta.get("output_dir", "")
    msa_path   = os.path.join(output_dir, f"{session_name}_msa.fasta")
    alignment_component = no_update
    if os.path.exists(msa_path):
        with open(msa_path) as f:
            aln_data = f.read()
        alignment_component = dashbio.AlignmentChart(
            id="alignment-chart", data=aln_data, height=420,
            tilewidth=20, showconservation=True, showgap=True,
        )
    plot = build_residue_plot(session_name, meta, slot=0)
    return session_name, \
           dbc.Alert(f"Loaded '{session_name}'.", color="success",
                     style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0}), \
           alignment_component, plot

@app.callback(
    Output("download-zip",    "data"),
    Output("download-status", "children"),
    Input("dl-zip-btn",       "n_clicks"),
    State("session-store",    "data"),
    prevent_initial_call=True,
)
def download_zip(n_clicks, session_name):
    if not session_name:
        return no_update, dbc.Alert("No active session.", color="warning",
                                    style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})
    meta    = db_hgetall(redis_session_key(session_name, "meta"))
    out_dir = meta.get("output_dir", "")
    if not out_dir or not os.path.exists(out_dir):
        return no_update, dbc.Alert("Output directory not found.", color="danger",
                                    style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})
    json_path = os.path.join(out_dir, f"{session_name}_summary.json")
    msa_path  = os.path.join(out_dir, f"{session_name}_msa.fasta")
    buf = io_module.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(json_path):
            zf.write(json_path, arcname=f"{session_name}_summary.json")
        if os.path.exists(msa_path):
            zf.write(msa_path,  arcname=f"{session_name}_msa.fasta")
    buf.seek(0)
    return dcc.send_bytes(buf.read(), filename=f"{session_name}_results.zip"), \
           dbc.Alert("Downloading…", color="info",
                     style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)