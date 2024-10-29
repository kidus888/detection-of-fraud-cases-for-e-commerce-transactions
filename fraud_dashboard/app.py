import pandas as pd
from flask import Flask, render_template
from dashboard.dashboard import create_dashboard  # Import create_dashboard function to create Dash app

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Initialize the Dash app with Flask
create_dashboard(app)

if __name__ == '__main__':
    app.run(debug=True)
