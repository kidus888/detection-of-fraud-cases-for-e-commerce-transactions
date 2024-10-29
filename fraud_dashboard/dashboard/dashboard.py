from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

def create_dashboard(server):
    # Load the data
    df = pd.read_csv('data/Fraud_Data.csv')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Initialize Dash app with Flask
    app = Dash(
        __name__,
        server=server,
        url_base_pathname='/dashboard/',
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )

    # Summary statistics
    def get_summary_stats(filtered_df):
        total_transactions = len(filtered_df)
        fraud_cases = filtered_df['class'].sum()
        fraud_percentage = (fraud_cases / total_transactions) * 100 if total_transactions > 0 else 0
        return total_transactions, fraud_cases, fraud_percentage

    total_transactions, fraud_cases, fraud_percentage = get_summary_stats(df)

    # Layout components
    app.layout = dbc.Container([
        html.H1("Fraud Detection Dashboard", className="text-center"),
        
        # Date range picker
        html.Br(),
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=df['purchase_time'].min(),
            max_date_allowed=df['purchase_time'].max(),
            start_date=df['purchase_time'].min(),
            end_date=df['purchase_time'].max()
        ),
        
        # Summary stats boxes
        html.Br(),
        dbc.Row([
            dbc.Col(html.Div(id="summary-stats"), width=12)
        ]),
        
        # Plots
        html.Br(),
        dbc.Row([
            dbc.Col(dcc.Graph(id="fraud-trend-line-chart"), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="fraud-by-device-bar-chart"), width=6),
            dbc.Col(dcc.Graph(id="fraud-by-browser-bar-chart"), width=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="fraud-by-age-gender"), width=6),
            dbc.Col(dcc.Graph(id="transaction-hour-heatmap"), width=6),
        ])
    ])

    # Callbacks for interactivity

    # Update summary stats
    @app.callback(
        Output("summary-stats", "children"),
        [Input("date-picker-range", "start_date"), Input("date-picker-range", "end_date")]
    )
    def update_summary(start_date, end_date):
        filtered_df = df[(df['purchase_time'] >= start_date) & (df['purchase_time'] <= end_date)]
        total_transactions, fraud_cases, fraud_percentage = get_summary_stats(filtered_df)
        return dbc.Row([
            dbc.Col(html.Div([html.H5("Total Transactions"), html.H4(total_transactions)])),
            dbc.Col(html.Div([html.H5("Fraud Cases"), html.H4(fraud_cases)])),
            dbc.Col(html.Div([html.H5("Fraud Percentage"), html.H4(f"{fraud_percentage:.2f}%")]))
        ])

    # Update fraud trend line chart based on date range
    @app.callback(
        Output("fraud-trend-line-chart", "figure"),
        [Input("date-picker-range", "start_date"), Input("date-picker-range", "end_date")]
    )
    def update_fraud_trend(start_date, end_date):
        filtered_df = df[(df['purchase_time'] >= start_date) & (df['purchase_time'] <= end_date)]
        fraud_trend = filtered_df[filtered_df['class'] == 1].groupby(filtered_df['purchase_time'].dt.date).size().reset_index(name="Fraud Cases")
        fig = px.line(fraud_trend, x="purchase_time", y="Fraud Cases", title="Fraud Cases Over Time")
        return fig

    # Update fraud by device and browser bar charts
    @app.callback(
        [Output("fraud-by-device-bar-chart", "figure"),
         Output("fraud-by-browser-bar-chart", "figure")],
        [Input("date-picker-range", "start_date"), Input("date-picker-range", "end_date")]
    )
    def update_device_browser_charts(start_date, end_date):
        filtered_df = df[(df['purchase_time'] >= start_date) & (df['purchase_time'] <= end_date) & (df['class'] == 1)]
        device_counts = filtered_df['device_id'].value_counts()
        browser_counts = filtered_df['browser'].value_counts()
        
        device_fig = go.Figure(data=[go.Bar(x=device_counts.index, y=device_counts.values)])
        device_fig.update_layout(title="Fraud Cases by Device")

        browser_fig = go.Figure(data=[go.Bar(x=browser_counts.index, y=browser_counts.values)])
        browser_fig.update_layout(title="Fraud Cases by Browser")
        
        return device_fig, browser_fig

    # Age and Gender fraud cases bar chart
    @app.callback(
        Output("fraud-by-age-gender", "figure"),
        [Input("date-picker-range", "start_date"), Input("date-picker-range", "end_date")]
    )
    def update_age_gender_chart(start_date, end_date):
        filtered_df = df[(df['purchase_time'] >= start_date) & (df['purchase_time'] <= end_date) & (df['class'] == 1)]
        age_gender_fig = px.histogram(filtered_df, x="age", color="sex", barmode="group", title="Fraud Cases by Age and Gender")
        return age_gender_fig

    # Heatmap of transactions by hour
    @app.callback(
        Output("transaction-hour-heatmap", "figure"),
        [Input("date-picker-range", "start_date"), Input("date-picker-range", "end_date")]
    )
    def update_transaction_hour_heatmap(start_date, end_date):
        filtered_df = df[(df['purchase_time'] >= start_date) & (df['purchase_time'] <= end_date)]
        filtered_df['hour'] = filtered_df['purchase_time'].dt.hour
        hour_counts = filtered_df[filtered_df['class'] == 1].groupby('hour').size().reset_index(name="Fraud Cases")
        
        fig = px.density_heatmap(hour_counts, x='hour', y='Fraud Cases', title="Fraud Cases by Hour")
        return fig

    return app
