# utils/plotting.py
import plotly.graph_objects as go
import pandas as pd
from utils.forecasting import forecast_series

def generate_time_series_plot(
    df, date_col, metric_col, freq,
    title, aggregation_type='sum',
    forecast_periods=4, run_forecast=False,
    is_duration=False
):
    if metric_col not in df.columns or date_col not in df.columns:
        return go.Figure()

    df = df.copy()
    df.set_index(date_col, inplace=True)

    if aggregation_type == 'sum':
        ts = df[metric_col].resample(freq).sum().fillna(0)
        yaxis = "Case Count"
    else:
        ts = df[metric_col].resample(freq).mean()
        yaxis = "Average Duration" if is_duration else "Average Value"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values,
        mode='lines+markers',
        name='Historical'
    ))

    if run_forecast:
        fc = forecast_series(ts, forecast_periods, freq)
        fig.add_trace(go.Scatter(
            x=fc['ds'], y=fc['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(dash='dash')
        ))

    fig.update_layout(
        title=title,
        yaxis_title=yaxis,
        height=350
    )
    return fig
