#!//usr/bin/python3

import os
from collections import Counter

from dash import Dash, Input, Output, State, callback, ctx, dcc, html
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_bio as dashbio
from dash_bio.utils import PdbParser as DashPdbParser, create_mol3d_style
import plotly.express as px
from Bio.PDB import PDBList, PDBParser, parse_pdb_header # type: ignore

import pandas as pd


# -------------------------
# Initialize the Dash app
# -------------------------
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash()
server = app.server


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050, debug=True)