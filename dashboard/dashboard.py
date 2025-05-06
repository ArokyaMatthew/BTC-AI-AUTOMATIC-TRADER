"""
Simple Dashboard for Bitcoin Trading Bot
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_loader import fetch_historical_data
from trading_bot import TradingBot

# Initialize app
app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css'])

app.layout = html.Div([
    html.H1('Bitcoin AI Trading Bot Dashboard'),
    
    html.Div([
        html.Div([
            html.H3('Trading Bot Status'),
            html.Div(id='bot-status'),
            html.Button('Start Bot', id='start-button', className='btn btn-success mr-2'),
            html.Button('Stop Bot', id='stop-button', className='btn btn-danger')
        ], className='col-md-4'),
        
        html.Div([
            html.H3('Performance Metrics'),
            html.Div(id='performance-metrics')
        ], className='col-md-8')
    ], className='row mb-4'),
    
    html.Div([
        html.H3('Price Chart and Signals'),
        dcc.DatePickerRange(
            id='date-range',
            start_date=(datetime.now() - timedelta(days=30)).date(),
            end_date=datetime.now().date()
        ),
        dcc.Graph(id='price-chart')
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=60 * 1000,  # in milliseconds (1 minute)
        n_intervals=0
    )
], className='container')

# Initialize bot globally
bot = None

@app.callback(
    [Output('bot-status', 'children'),
     Output('performance-metrics', 'children'),
     Output('price-chart', 'figure')],
    [Input('interval-component', 'n_intervals'),
     Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_dashboard(n_intervals, start_clicks, stop_clicks, start_date, end_date):
    global bot
    
    ctx = dash.callback_context
    
    # Initialize bot if not already initialized
    if bot is None:
        bot = TradingBot(strategy_name=config.STRATEGY, testnet=config.USE_TESTNET)
        bot.load_state()
    
    # Handle button clicks
    if ctx.triggered and 'start-button' in ctx.triggered[0]['prop_id'] and start_clicks:
        bot.run()
    
    if ctx.triggered and 'stop-button' in ctx.triggered[0]['prop_id'] and stop_clicks:
        bot.stop()
    
    # Get bot status
    status_elements = []
    if bot.running:
        status = html.Div([
            html.H4('Running', style={'color': 'green'}),
            html.P(f"Strategy: {bot.strategy_name}"),
            html.P(f"Current Position: {'Long' if bot.current_position > 0 else 'Short' if bot.current_position < 0 else 'None'}")
        ])
    else:
        status = html.Div([
            html.H4('Stopped', style={'color': 'red'}),
            html.P(f"Strategy: {bot.strategy_name}")
        ])
    
    status_elements.append(status)
    
    # Get performance metrics
    metrics = bot.get_performance_metrics()
    metrics_elements = html.Div([
        html.Table([
            html.Tr([html.Td('Total Trades:'), html.Td(f"{metrics['total_trades']}")]),
            html.Tr([html.Td('Win Rate:'), html.Td(f"{metrics['win_rate']:.2%}")]),
            html.Tr([html.Td('Total Profit:'), html.Td(f"${metrics['total_profit']:.2f}")]),
            html.Tr([html.Td('Avg Profit per Trade:'), html.Td(f"${metrics['avg_profit_per_trade']:.2f}")]),
            html.Tr([html.Td('Max Profit:'), html.Td(f"${metrics['max_profit']:.2f}")]),
            html.Tr([html.Td('Max Loss:'), html.Td(f"${metrics['max_loss']:.2f}")])
        ], className='table table-striped')
    ])
    
    # Get price chart
    if start_date and end_date:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Fetch historical data
        df = fetch_historical_data(
            symbol=config.SYMBOL,
            timeframe=config.TIMEFRAME,
            start_date=start,
            end_date=end
        )
        
        if not df.empty:
            # Generate signals
            signals_df = bot.strategy.generate_signals(df)
            
            # Create price chart
            fig = go.Figure()
            
            # Add price candles
            fig.add_trace(go.Candlestick(
                x=signals_df.index,
                open=signals_df['open'],
                high=signals_df['high'],
                low=signals_df['low'],
                close=signals_df['close'],
                name='Price'
            ))
            
            # Add buy signals
            buy_signals = signals_df[signals_df['signal'] > 0]
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['low'] * 0.99,  # Offset for visibility
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green'
                ),
                name='Buy Signal'
            ))
            
            # Add sell signals
            sell_signals = signals_df[signals_df['signal'] < 0]
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['high'] * 1.01,  # Offset for visibility
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red'
                ),
                name='Sell Signal'
            ))
            
            # Layout
            fig.update_layout(
                title=f"{config.SYMBOL} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=False,
                height=600
            )
        else:
            # Empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title="No data available",
                height=600
            )
    else:
        # Empty figure if no date range selected
        fig = go.Figure()
        fig.update_layout(
            title="Select date range",
            height=600
        )
    
    return status_elements, metrics_elements, fig

if __name__ == '__main__':
    app.run_server(debug=config.DEBUG, host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT)