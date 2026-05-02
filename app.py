#!/usr/bin/python3
import os
import base64
import json
import subprocess
import tempfile
import numpy as np
from collections import Counter

import redis
from dash import Dash, Input, Output, State, callback, ctx, dcc, html, no_update, ALL
import dash_bootstrap_components as dbc
import dash_bio as dashbio
from dash_bio.utils import PdbParser as DashPdbParser, create_mol3d_style
import plotly.graph_objects as go

# ── Constants ─────────────────────────────────────────────────────────────────
REDIS_HOST = os.environ.get("REDIS_HOST", "redis-staging")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
MUSCLE_PATH = "/code/MUSCLE/muscle-linux-x86.v5.3"
P2RANK_PATH = "/code/p2rank_2.5.1"
SESSION_BASE = "/tmp/sessions"

# ── Redis client ──────────────────────────────────────────────────────────────
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

try:
    r.ping()
except redis.exceptions.ConnectionError as e:
    raise RuntimeError(f"Redis connection failed: {e}")

# ── Initialize ────────────────────────────────────────────────────────────────
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# ── Helpers ───────────────────────────────────────────────────────────────────
def gaussian_curve(mean, std, x_range=(0, 100), n=300):
    x = np.linspace(*x_range, n)
    y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    return x, y

def make_gaussian_fig(mean, std):
    x, y = gaussian_curve(mean, std)
    y_norm = y / y.max() if y.max() > 0 else y
    mask = (x >= mean - std) & (x <= mean + std)
    shade_x = x[mask]
    shade_y = y_norm[mask]
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

def redis_session_key(session_name, suffix):
    return f"session:{session_name}:{suffix}"

def store_file_in_redis(session_name, slot, filename, contents_b64):
    """Store uploaded file (base64) and filename in Redis."""
    key = redis_session_key(session_name, f"file{slot}")
    r.hset(key, mapping={"filename": filename, "contents": contents_b64})
    r.expire(key, 60 * 60 * 24)  # 24h TTL

def write_file_to_disk(session_dir, filename, contents_b64):
    """Decode base64 upload and write to session input dir. Returns path."""
    _, content_string = contents_b64.split(",")
    decoded = base64.b64decode(content_string)
    path = os.path.join(session_dir, filename)
    with open(path, "wb") as f:
        f.write(decoded)
    return path

# ── Slider config ─────────────────────────────────────────────────────────────
SLIDERS = [
    ("seq-weight",    "Sequence conservation weight (a)", 0.0, 2.0, 0.01, 0.3),
    ("plddt-weight",  "pLDDT weight (b)",                 0.0, 2.0, 0.01, 0.2),
    ("p2rank-weight", "P2Rank weight (c)",                0.0, 2.0, 0.01, 0.5),
    ("gauss-mean",    "Gaussian mean (pLDDT)",            0,   100, 1,    70),
    ("gauss-std",     "Gaussian std dev",                 1,   30,  0.5,  5),
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

# ── Layout ────────────────────────────────────────────────────────────────────
CARD = {"border": "0.5px solid #dee2e6", "borderRadius": "12px", "backgroundColor": "white"}

app.layout = dbc.Container(fluid=True, style={"backgroundColor": "#f5f5f3", "minHeight": "100vh", "padding": "1.5rem"}, children=[

    # Header
    html.Div(style={"marginBottom": "1.5rem"}, children=[
        html.H1("Protein Binding Site Analysis", style={
            "fontFamily": "'Georgia', serif", "fontWeight": "400",
            "fontSize": "26px", "marginBottom": "2px", "letterSpacing": "-0.3px"
        }),
        html.P("Structure-based scoring pipeline", style={"color": "#888", "fontSize": "13px", "margin": 0}),
    ]),

    # ── Row 1: Submission + Parameters + Gaussian ─────────────────────────────
    dbc.Row(className="g-3 mb-3", children=[

        # Submission Panel
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

                # File 1
                html.Label("Structure 1", style={"fontSize": "12px", "color": "#6c757d", "marginBottom": "3px"}),
                dcc.Upload(id="upload-file1", accept=".cif,.pdb", multiple=False,
                    children=html.Div(id="upload-file1-label", children=[
                        html.Span("↑ ", style={"color": "#ccc"}),
                        html.Span("Click or drag .pdb/.cif", style={"fontSize": "12px", "color": "#999"}),
                    ]),
                    style={"border": "1.5px dashed #ddd", "borderRadius": "8px",
                           "padding": "0.6rem 1rem", "cursor": "pointer", "marginBottom": "0.75rem"}
                ),

                # File 2
                html.Label("Structure 2", style={"fontSize": "12px", "color": "#6c757d", "marginBottom": "3px"}),
                dcc.Upload(id="upload-file2", accept=".cif,.pdb", multiple=False,
                    children=html.Div(id="upload-file2-label", children=[
                        html.Span("↑ ", style={"color": "#ccc"}),
                        html.Span("Click or drag .pdb/.cif", style={"fontSize": "12px", "color": "#999"}),
                    ]),
                    style={"border": "1.5px dashed #ddd", "borderRadius": "8px",
                           "padding": "0.6rem 1rem", "cursor": "pointer", "marginBottom": "0.75rem"}
                ),

                # Additional files
                html.Label("Additional structures (optional)", style={"fontSize": "12px", "color": "#6c757d", "marginBottom": "3px"}),
                dcc.Upload(id="upload-extra", accept=".cif,.pdb", multiple=True,
                    children=html.Div([
                        html.Span("↑ ", style={"color": "#ccc"}),
                        html.Span("Click or drag multiple files", style={"fontSize": "12px", "color": "#999"}),
                    ]),
                    style={"border": "1.5px dashed #ddd", "borderRadius": "8px",
                           "padding": "0.6rem 1rem", "cursor": "pointer", "marginBottom": "1rem"}
                ),

                html.Div(id="file-status", style={"marginBottom": "0.75rem", "minHeight": "24px"}),

                html.Hr(style={"borderColor": "#eee", "margin": "0.75rem 0"}),

                dbc.Button("Run analysis", id="run-btn", color="primary", size="sm",
                           disabled=True,
                           style={"width": "100%", "borderRadius": "8px",
                                  "fontWeight": "500", "fontSize": "13px"}),
                html.Div(id="run-status", style={"marginTop": "0.5rem", "minHeight": "24px"}),
            ])])
        ]),

        # Parameters Panel
        dbc.Col(width=5, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("Scoring weights", style={"fontWeight": "500", "marginBottom": "1rem", "fontSize": "13px",
                                                   "textTransform": "uppercase", "letterSpacing": "0.05em", "color": "#888"}),
                *[make_slider_row(*s) for s in SLIDERS],
            ])])
        ]),

        # Gaussian Preview
        dbc.Col(width=3, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("pLDDT gaussian weight", style={"fontWeight": "500", "marginBottom": "0.5rem",
                                                         "fontSize": "13px", "textTransform": "uppercase",
                                                         "letterSpacing": "0.05em", "color": "#888"}),
                html.P("Residues with pLDDT near the mean receive higher weight. "
                       "Shaded band shows ±1 std dev.",
                       style={"fontSize": "11px", "color": "#999", "marginBottom": "0.75rem", "lineHeight": "1.5"}),
                dcc.Graph(id="gaussian-plot", figure=make_gaussian_fig(70, 5),
                          config={"displayModeBar": False}, style={"margin": "0 -8px"}),
                html.Div(id="gaussian-stats", style={"marginTop": "0.5rem"}),
            ])])
        ]),
    ]),

    # ── Row 2: Molecule Viewer + Alignment Viewer ─────────────────────────────
    dbc.Row(className="g-3", children=[

        dbc.Col(width=6, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("Structure viewer", style={"fontWeight": "500", "marginBottom": "0.75rem",
                                                    "fontSize": "13px", "textTransform": "uppercase",
                                                    "letterSpacing": "0.05em", "color": "#888"}),

                # File selector tabs when multiple structures loaded
                html.Div(id="mol3d-selector", style={"marginBottom": "0.5rem"}),

                html.Div(id="mol3d-container", style={"minHeight": "420px"}, children=[
                    html.Div(style={
                        "height": "420px", "display": "flex", "alignItems": "center",
                        "justifyContent": "center", "color": "#ccc", "fontSize": "13px",
                        "border": "1px dashed #eee", "borderRadius": "8px"
                    }, children="Upload structures to view")
                ]),
            ])])
        ]),

        dbc.Col(width=6, children=[
            dbc.Card(style=CARD, children=[dbc.CardBody(style={"padding": "1.25rem"}, children=[
                html.H6("Sequence alignment", style={"fontWeight": "500", "marginBottom": "0.75rem",
                                                      "fontSize": "13px", "textTransform": "uppercase",
                                                      "letterSpacing": "0.05em", "color": "#888"}),
                html.Div(id="alignment-container", style={"minHeight": "420px"}, children=[
                    html.Div(style={
                        "height": "420px", "display": "flex", "alignItems": "center",
                        "justifyContent": "center", "color": "#ccc", "fontSize": "13px",
                        "border": "1px dashed #eee", "borderRadius": "8px"
                    }, children="Alignment will appear after analysis runs")
                ]),
            ])])
        ]),
    ]),

    # Polling interval — checks Redis for job completion every 3s
    dcc.Interval(id="poll-interval", interval=3000, n_intervals=0, disabled=True),

    # Stores
    dcc.Store(id="file1-store"),
    dcc.Store(id="file2-store"),
    dcc.Store(id="extra-store"),
    dcc.Store(id="session-store"),        # active session name
    dcc.Store(id="mol3d-active", data=1), # which file slot is displayed
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

# Slider value displays — one callback per slider to avoid closure issues
@app.callback(Output("val-seq-weight",   "children"), Input("slider-seq-weight",   "value"))
def _v0(v): return f"{v:.2f}"

@app.callback(Output("val-plddt-weight", "children"), Input("slider-plddt-weight", "value"))
def _v1(v): return f"{v:.2f}"

@app.callback(Output("val-p2rank-weight","children"), Input("slider-p2rank-weight","value"))
def _v2(v): return f"{v:.2f}"

@app.callback(Output("val-gauss-mean",   "children"), Input("slider-gauss-mean",   "value"))
def _v3(v): return f"{v:.0f}"

@app.callback(Output("val-gauss-std",    "children"), Input("slider-gauss-std",    "value"))
def _v4(v): return f"{v:.1f}"


@app.callback(
    Output("gaussian-plot",  "figure"),
    Output("gaussian-stats", "children"),
    Input("slider-gauss-mean", "value"),
    Input("slider-gauss-std",  "value"),
)
def update_gaussian(mean, std):
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


# File upload handlers — store in dcc.Store and show filename label
@app.callback(
    Output("file1-store",        "data"),
    Output("upload-file1-label", "children"),
    Input("upload-file1",        "contents"),
    State("upload-file1",        "filename"),
    prevent_initial_call=True,
)
def store_file1(contents, filename):
    if not contents:
        return no_update, no_update
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdb", ".cif"]:
        return None, dbc.Badge("Invalid file type", color="danger", style={"fontSize": "11px"})
    label = [
        html.Span("✓ ", style={"color": "#28a745", "fontSize": "12px"}),
        html.Span(filename, style={"fontSize": "12px", "color": "#495057"}),
    ]
    return {"filename": filename, "contents": contents}, label


@app.callback(
    Output("file2-store",        "data"),
    Output("upload-file2-label", "children"),
    Input("upload-file2",        "contents"),
    State("upload-file2",        "filename"),
    prevent_initial_call=True,
)
def store_file2(contents, filename):
    if not contents:
        return no_update, no_update
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdb", ".cif"]:
        return None, dbc.Badge("Invalid file type", color="danger", style={"fontSize": "11px"})
    label = [
        html.Span("✓ ", style={"color": "#28a745", "fontSize": "12px"}),
        html.Span(filename, style={"fontSize": "12px", "color": "#495057"}),
    ]
    return {"filename": filename, "contents": contents}, label


@app.callback(
    Output("extra-store", "data"),
    Input("upload-extra", "contents"),
    State("upload-extra", "filename"),
    prevent_initial_call=True,
)
def store_extra(contents_list, filenames):
    if not contents_list:
        return no_update
    valid = [
        {"filename": fn, "contents": c}
        for fn, c in zip(filenames, contents_list)
        if os.path.splitext(fn)[1].lower() in [".pdb", ".cif"]
    ]
    return valid or None


# Enable/disable run button and show file status
@app.callback(
    Output("run-btn",     "disabled"),
    Output("file-status", "children"),
    Input("session-name", "value"),
    Input("file1-store",  "data"),
    Input("file2-store",  "data"),
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
    Output("run-status",    "children"),
    Output("session-store", "data"),
    Output("poll-interval", "disabled"),
    Input("run-btn",             "n_clicks"),
    State("session-name",        "value"),
    State("file1-store",         "data"),
    State("file2-store",         "data"),
    State("extra-store",         "data"),
    State("slider-seq-weight",   "value"),
    State("slider-plddt-weight", "value"),
    State("slider-p2rank-weight","value"),
    State("slider-gauss-mean",   "value"),
    State("slider-gauss-std",    "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, session_name, file1, file2, extra,
                 w_seq, w_plddt, w_p2rank, g_mean, g_std):

    session_name = session_name.strip()

    # Create session dirs
    session_dir  = os.path.join(SESSION_BASE, session_name)
    input_dir    = os.path.join(session_dir, "Input")
    work_dir     = os.path.join(session_dir, "Work")
    output_dir   = os.path.join(session_dir, "Output")
    for d in [input_dir, work_dir, output_dir]:
        os.makedirs(d, exist_ok=True)

    # Write uploaded files to input dir + store in Redis
    all_files = [file1, file2] + (extra or [])
    for i, f in enumerate(all_files, start=1):
        path = write_file_to_disk(input_dir, f["filename"], f["contents"])
        store_file_in_redis(session_name, i, f["filename"], f["contents"])

    # Store session metadata in Redis
    r.hset(redis_session_key(session_name, "meta"), mapping={
        "status": "running",
        "session_name": session_name,
        "output_dir": output_dir,
        "file_count": len(all_files),
    })
    r.expire(redis_session_key(session_name, "meta"), 60 * 60 * 24)

    # Launch calc_stats.py as subprocess
    cmd = [
        "python3", "/code/main_workflow/calc_stats.py",
        "-n", session_name,
        "-m", MUSCLE_PATH,
        "-p", P2RANK_PATH,
        "-i", input_dir,
        "-w", work_dir,
        "-o", output_dir,
        "-a", str(w_seq),
        "-b", str(w_plddt),
        "-c", str(w_p2rank),
        "-u", str(g_mean),
        "-s", str(g_std),
        "-l", "INFO",
    ]

    # ── Output parsing helper ─────────────────────────────────────────────────────

    def load_session_results(session_name: str, output_dir: str) -> dict | None:
        """Parse the summary JSON and push per-protein scores into Redis."""
        summary_path = os.path.join(output_dir, f"{session_name}_summary.json")
        if not os.path.exists(summary_path):
            return None

        with open(summary_path) as f:
            root = json.load(f)

        result = root["result"]

        # Push per-protein residue scores into Redis keyed by filename
        for protein in result["protein_files"]:
            fname = protein["file_name"]
            scores = protein["final_score_per_residue"]  # list indexed by residue position
            score_key = redis_session_key(session_name, f"scores:{fname}")
            # Store as {residue_index_str: score} for fast lookup
            r.set(score_key, json.dumps({str(i): s for i, s in enumerate(scores)}))
            r.expire(score_key, 60 * 60 * 24)

        # Push session-level conservation scores
        r.hset(redis_session_key(session_name, "meta"), mapping={
            "status": "complete",
            "output_dir": output_dir,
            "final_score": json.dumps(result["final_score"]),
            "sequence_conservation": json.dumps(result["sequence_conservation"]),
            "plddt_conservation": json.dumps(result["PLDDT_score_conservation"]),
            "p2rank_conservation": json.dumps(result["P2Rank_score_conservation"]),
        })

        return result

    def run_and_update_redis():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                parsed = load_session_results(session_name, output_dir)
                if parsed is None:
                    r.hset(redis_session_key(session_name, "meta"),
                        mapping={"status": "error",
                                    "stderr": "Output JSON not found after completion."})
            else:
                r.hset(redis_session_key(session_name, "meta"),
                    mapping={"status": "error", "stderr": result.stderr[-2000:]})
        except Exception as e:
            r.hset(redis_session_key(session_name, "meta"),
                mapping={"status": "error", "stderr": str(e)})

    import threading
    threading.Thread(target=run_and_update_redis, daemon=True).start()

    msg = dbc.Alert([
                html.Span(
                dbc.Spinner(size="sm", color="primary"),
                style={"marginRight": "8px"}
                        ),
        f"Session '{session_name}' running…"
    ], color="info", style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})

    return msg, session_name, False


@app.callback(
    Output("run-status",           "children",  allow_duplicate=True),
    Output("poll-interval",        "disabled",  allow_duplicate=True),
    Output("alignment-container",  "children"),
    Output("mol3d-container",      "children",  allow_duplicate=True),
    Input("poll-interval",         "n_intervals"),
    State("session-store",         "data"),
    State("file1-store",           "data"),
    State("file2-store",           "data"),
    State("extra-store",           "data"),
    State("mol3d-active",          "data"),
    prevent_initial_call=True,
)
def poll_results(n_intervals, session_name, file1, file2, extra, active_slot):
    if not session_name:
        return no_update, True, no_update, no_update

    meta   = r.hgetall(redis_session_key(session_name, "meta"))
    status = meta.get("status", "unknown")

    if status == "running":
        return no_update, False, no_update, no_update

    if status == "error":
        err = meta.get("stderr", "Unknown error")
        msg = dbc.Alert(f"Analysis failed: {err[:300]}", color="danger",
                        style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})
        return msg, True, no_update, no_update

    if status == "complete":
        output_dir  = meta.get("output_dir", "")
        status_msg  = dbc.Alert("Analysis complete.", color="success",
                                style={"fontSize": "12px", "padding": "6px 10px", "marginBottom": 0})

        # ── Alignment viewer ──────────────────────────────────────────────
        msa_path = os.path.join(output_dir, f"{session_name}_msa.fasta")
        alignment_component = no_update
        if os.path.exists(msa_path):
            with open(msa_path) as f:
                aln_data = f.read()
            alignment_component = dashbio.AlignmentChart(
                id="alignment-chart",
                data=aln_data,
                height=420,
                tilewidth=20,
                showconservation=True,
                showgap=True,
            )

def render_mol3d(file_data: dict, session_name: str | None = None):
    """Render a Molecule3dViewer, optionally coloring residues by analysis scores."""
    try:
        _, content_string = file_data["contents"].split(",")
        decoded = base64.b64decode(content_string)
        suffix  = os.path.splitext(file_data["filename"])[1]

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name

        parser   = DashPdbParser(tmp_path)
        mol_data = parser.mol3d_data()
        os.unlink(tmp_path)

        styles = create_mol3d_style(
            mol_data["atoms"],
            visualization_type="cartoon",
            color_element="residue_type"
        )

        # Color by final_score_per_residue if available in Redis
        if session_name:
            score_key   = redis_session_key(session_name, f"scores:{file_data['filename']}")
            scores_json = r.get(score_key)
            if scores_json:
                scores     = json.loads(scores_json)
                serial_map = {atom["serial"]: str(i)
                              for i, atom in enumerate(mol_data["atoms"])}

                for style in styles:
                    res_idx = serial_map.get(style.get("serial"), "0")
                    score   = float(scores.get(res_idx, 0.5))
                    # Blue (low) → White (mid) → Red (high)
                    if score >= 0.5:
                        t     = (score - 0.5) * 2        # 0→1
                        r_val = 255
                        g_val = int(255 * (1 - t))
                        b_val = int(255 * (1 - t))
                    else:
                        t     = score * 2                 # 0→1
                        r_val = int(255 * t)
                        g_val = int(255 * t)
                        b_val = 255
                    style["color"] = f"#{r_val:02x}{g_val:02x}{b_val:02x}"

        return dashbio.Molecule3dViewer(
            id="mol3d-viewer",
            modelData=mol_data,
            styles=styles,
            height=420,
            backgroundColor="#fafafa",
        )

    except Exception as e:
        return dbc.Alert(f"Error loading structure: {e}", color="danger",
                         style={"fontSize": "12px", "margin": 0})


# Update update_mol3d callback to use render_mol3d
@app.callback(
    Output("mol3d-container", "children"),
    Output("mol3d-selector",  "children"),
    Input("file1-store",      "data"),
    Input("file2-store",      "data"),
    Input("extra-store",      "data"),
    Input("mol3d-active",     "data"),
    State("session-store",    "data"),
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
        dbc.Button(
            f["filename"], id={"type": "mol3d-btn", "index": i + 1},
            size="sm", outline=(i + 1 != active_slot),
            color="primary",
            style={"fontSize": "11px", "borderRadius": "6px", "padding": "2px 10px"}
        )
        for i, f in enumerate(all_files)
    ])

    idx         = min((active_slot or 1) - 1, len(all_files) - 1)
    active_file = all_files[idx]
    viewer      = render_mol3d(active_file, session_name)

    return viewer, selector


@app.callback(
    Output("mol3d-active", "data"),
    Input({"type": "mol3d-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def switch_mol3d(n_clicks):
    if not any(n_clicks):
        return no_update
    triggered = ctx.triggered_id
    return triggered["index"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)