#!//usr/bin/python3
from dash import Dash, html, dcc

app = Dash()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050, debug=True)