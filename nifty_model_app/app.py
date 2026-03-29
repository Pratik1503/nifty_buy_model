import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import streamlit as st

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Streamlit App Title
st.set_page_config(layout="wide")
st.title('NIFTY 50 Market Timing Strategy Backtest')
st.markdown("--- ")


# --- Helper Functions ---
def flatten_cols(df):
    cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            cols.append(col[0])
        else:
            cols.append(col)
    df.columns = cols
    return df

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def normalize_series(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index) # Handle cases with no variation
    return (series - min_val) / (max_val - min_val)

def apply_position_sizing(df_input):
    df_temp = df_input.copy()
    df_temp['Position'] = 0.0
    df_temp.loc[df_temp['Entry_Score'] > 0.6, 'Position'] = 1.0
    df_temp.loc[(df_temp['Entry_Score'] >= 0.3) & (df_temp['Entry_Score'] <= 0.6), 'Position'] = 0.5
    df_temp.loc[df_temp['Entry_Score'] < 0.3, 'Position'] = 0.15
    return df_temp


# --- 1. DATA: Download NIFTY 50 Data ---
st.header('Step 1: Data Download')
ticker = '^NSEI'
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=8 * 365) # Approximate 8 years

st.write(f"Downloading {ticker} data from {start_date} to {end_date}...")
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

df = load_data(ticker, start_date, end_date)
st.write(f"Downloaded {len(df)} days of data for {ticker}.")

# Flatten columns immediately after download
df = flatten_cols(df.copy())


# --- 2. FEATURES: Calculate Technical Indicators ---
st.header('Step 2: Feature Engineering')
with st.spinner('Calculating technical indicators...'):
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Momentum_Strength'] = (df['Close'] / df['MA200']) - 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_20D'] = df['Daily_Return'].rolling(window=20).std()
    df['Rolling_Max'] = df['High'].rolling(window=252).max()
    df['Drawdown'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max']
    df['RSI'] = calculate_rsi(df['Close'])

    df.dropna(inplace=True)
st.success('Technical indicators calculated.')
st.dataframe(df.head())


# --- 3. ENTRY SCORE: Create a Composite Entry Score ---
st.header('Step 3: Interactive Entry Score Configuration')

# Calculate raw scores (independent of weights)
df['Momentum_Score'] = normalize_series(df['Momentum_Strength'])
df['Volatility_Score'] = 1 - normalize_series(df['Volatility_20D'])
df['Drawdown_Score'] = 1 - normalize_series(df['Drawdown'])
df['RSI_Score'] = 1 - normalize_series(df['RSI'])

st.markdown("Adjust the sliders below to change the weights of each component in the Entry Score. Weights will be auto-normalized.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    momentum_weight = st.slider('Momentum Weight', 0.0, 1.0, 0.25, 0.01)
with col2:
    volatility_weight = st.slider('Volatility Weight', 0.0, 1.0, 0.25, 0.01)
with col3:
    drawdown_weight = st.slider('Drawdown Weight', 0.0, 1.0, 0.25, 0.01)
with col4:
    rsi_weight = st.slider('RSI Weight', 0.0, 1.0, 0.25, 0.01)

# Auto-normalize weights
total_weight = momentum_weight + volatility_weight + drawdown_weight + rsi_weight
if total_weight == 0:
    norm_momentum_weight = 0.25
    norm_volatility_weight = 0.25
    norm_drawdown_weight = 0.25
    norm_rsi_weight = 0.25
else:
    norm_momentum_weight = momentum_weight / total_weight
    norm_volatility_weight = volatility_weight / total_weight
    norm_drawdown_weight = drawdown_weight / total_weight
    norm_rsi_weight = rsi_weight / total_weight

st.write(f"Effective Weights: Momentum={norm_momentum_weight:.2f}, Volatility={norm_volatility_weight:.2f}, Drawdown={norm_drawdown_weight:.2f}, RSI={norm_rsi_weight:.2f}")

df['Entry_Score'] = (
    df['Momentum_Score'] * norm_momentum_weight +
    df['Volatility_Score'] * norm_volatility_weight +
    df['Drawdown_Score'] * norm_drawdown_weight +
    df['RSI_Score'] * norm_rsi_weight
)

st.subheader('Entry Score Distribution')
st.write(df['Entry_Score'].describe())


# --- 4. POSITION SIZING STRATEGY ---
st.header('Step 4: Position Sizing Strategy')
st.write("Applying position sizing based on Entry Score thresholds:")
st.markdown("""
-   **High Score (above 0.6)**: 100% invested
-   **Medium Score (0.3-0.6)**: 50% invested
-   **Low Score (below 0.3)**: 15% invested
""")

df = apply_position_sizing(df)

st.subheader('Position Distribution')
st.dataframe(df[['Close', 'Entry_Score', 'Position']].tail())
st.write(df['Position'].value_counts(normalize=True))


# --- 5. BACKTEST ---
st.header('Step 5: Backtest Simulation')
initial_capital = 100000
st.write(f"Simulating backtest with initial capital: INR {initial_capital:,}")

with st.spinner('Running backtest...'):
    df['Market_Daily_Return'] = df['Close'].pct_change()
    df['Previous_Position'] = df['Position'].shift(1)
    df['Transaction_Cost_Amount'] = 0.0

    df['Strategy_Portfolio_Value'] = np.nan
    df['Strategy_Portfolio_Value_No_Cost'] = np.nan
    df['Buy_Hold_Portfolio_Value'] = np.nan

    first_valid_index = df['Previous_Position'].first_valid_index()

    if first_valid_index is not None:
        start_date_backtest = first_valid_index
        start_index_loc = df.index.get_loc(start_date_backtest)

        df.loc[start_date_backtest, 'Strategy_Portfolio_Value'] = initial_capital
        df.loc[start_date_backtest, 'Strategy_Portfolio_Value_No_Cost'] = initial_capital
        df.loc[start_date_backtest, 'Buy_Hold_Portfolio_Value'] = initial_capital

        for i in range(start_index_loc + 1, len(df)):
            current_date = df.index[i]
            previous_date = df.index[i-1]

            prev_strategy_value_with_cost = df.loc[previous_date, 'Strategy_Portfolio_Value']
            prev_strategy_value_no_cost = df.loc[previous_date, 'Strategy_Portfolio_Value_No_Cost']

            market_return_today = df.loc[current_date, 'Market_Daily_Return']
            current_position = df.loc[current_date, 'Position']
            prev_position = df.loc[current_date, 'Previous_Position']

            strategy_daily_return_based_on_prev_pos = market_return_today * prev_position

            df.loc[current_date, 'Strategy_Portfolio_Value_No_Cost'] = prev_strategy_value_no_cost * (1 + strategy_daily_return_based_on_prev_pos)

            transaction_cost_amount = 0.0
            if abs(current_position - prev_position) > 0.1:
                transaction_cost_amount = 0.001 * (prev_strategy_value_with_cost * (1 + strategy_daily_return_based_on_prev_pos))

            df.loc[current_date, 'Strategy_Portfolio_Value'] = (prev_strategy_value_with_cost * (1 + strategy_daily_return_based_on_prev_pos)) - transaction_cost_amount
            df.loc[current_date, 'Transaction_Cost_Amount'] = transaction_cost_amount

            prev_buy_hold_value = df.loc[previous_date, 'Buy_Hold_Portfolio_Value']
            df.loc[current_date, 'Buy_Hold_Portfolio_Value'] = prev_buy_hold_value * (1 + market_return_today)

df.dropna(subset=['Strategy_Portfolio_Value', 'Strategy_Portfolio_Value_No_Cost', 'Buy_Hold_Portfolio_Value'], inplace=True)

df['Strategy_Daily_Return'] = df['Strategy_Portfolio_Value'].pct_change()
df['Strategy_Daily_Return_No_Cost'] = df['Strategy_Portfolio_Value_No_Cost'].pct_change()
st.success('Backtest complete.')
st.dataframe(df[['Close', 'Position', 'Strategy_Portfolio_Value', 'Buy_Hold_Portfolio_Value']].tail())


# --- 6. VISUALS ---
st.header('Step 6: Visualizations')

# Plot 1: Portfolio growth
st.subheader('Portfolio Growth: Strategy vs Buy & Hold')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df['Strategy_Portfolio_Value'], mode='lines', name='Strategy Portfolio (With Costs)', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=df.index, y=df['Buy_Hold_Portfolio_Value'], mode='lines', name='Buy & Hold Portfolio', line=dict(color='red')))
fig1.update_layout(
    title_text='Portfolio Growth: Strategy (With Costs) vs Buy & Hold',
    xaxis_title='Date',
    yaxis_title='Portfolio Value (INR)',
    hovermode='x unified',
    height=500,
    template='plotly_white'
)
st.plotly_chart(fig1, use_container_width=True)

# Plot 2: Entry Score over time
st.subheader('Entry Score Over Time with Thresholds')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df['Entry_Score'], mode='lines', name='Entry Score', line=dict(color='purple')))
fig2.add_hline(y=0.6, line_dash='dot', line_color='green', annotation_text='High Score Threshold (0.6)', annotation_position='bottom right')
fig2.add_hline(y=0.3, line_dash='dot', line_color='red', annotation_text='Low Score Threshold (0.3)', annotation_position='top left')
fig2.update_layout(
    title_text='Entry Score Over Time with Thresholds',
    xaxis_title='Date',
    yaxis_title='Entry Score',
    hovermode='x unified',
    height=500,
    template='plotly_white'
)
st.plotly_chart(fig2, use_container_width=True)

# Plot 3: Drawdown chart
st.subheader('Drawdown: Strategy vs Buy & Hold')
df['Strategy_Daily_Returns_Drawdown'] = df['Strategy_Portfolio_Value'].pct_change()
df['Buy_Hold_Daily_Returns_Drawdown'] = df['Buy_Hold_Portfolio_Value'].pct_change()
df['Strategy_Cumulative_Returns'] = (1 + df['Strategy_Daily_Returns_Drawdown']).cumprod()
df['Buy_Hold_Cumulative_Returns'] = (1 + df['Buy_Hold_Daily_Returns_Drawdown']).cumprod()
df['Strategy_Rolling_Max_Cumulative'] = df['Strategy_Cumulative_Returns'].cummax()
df['Buy_Hold_Rolling_Max_Cumulative'] = df['Buy_Hold_Cumulative_Returns'].cummax()
df['Strategy_Drawdown'] = (df['Strategy_Cumulative_Returns'] / df['Strategy_Rolling_Max_Cumulative']) - 1
df['Buy_Hold_Drawdown'] = (df['Buy_Hold_Cumulative_Returns'] / df['Buy_Hold_Rolling_Max_Cumulative']) - 1

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df.index, y=df['Strategy_Drawdown'], mode='lines', name='Strategy Drawdown (With Costs)', fill='tozeroy', fillcolor='rgba(0,0,255,0.2)', line=dict(color='blue')))
fig3.add_trace(go.Scatter(x=df.index, y=df['Buy_Hold_Drawdown'], mode='lines', name='Buy & Hold Drawdown', fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', line=dict(color='red')))
fig3.update_layout(
    title_text='Drawdown: Strategy (With Costs) vs Buy & Hold',
    xaxis_title='Date',
    yaxis_title='Drawdown (%)',
    hovermode='x unified',
    height=500,
    template='plotly_white',
    yaxis_tickformat='.1%'
)
st.plotly_chart(fig3, use_container_width=True)

# Plot 4: NIFTY price with Buy Zones
st.subheader('NIFTY 50 Price with High Entry Score Buy Zones')
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='NIFTY 50 Close Price', line=dict(color='grey')))
high_score_mask = (df['Entry_Score'] > 0.6)

start_shade = None
for i in range(len(df)):
    if high_score_mask.iloc[i] and start_shade is None:
        start_shade = df.index[i]
    elif not high_score_mask.iloc[i] and start_shade is not None:
        fig4.add_vrect(x0=start_shade, x1=df.index[i], fillcolor='green', opacity=0.2, layer='below', line_width=0, name='High Score Period')
        start_shade = None
if start_shade is not None:
    fig4.add_vrect(x0=start_shade, x1=df.index[-1], fillcolor='green', opacity=0.2, layer='below', line_width=0, name='High Score Period')

fig4.update_layout(
    title_text='NIFTY 50 Price with High Entry Score Buy Zones',
    xaxis_title='Date',
    yaxis_title='Close Price',
    hovermode='x unified',
    height=500,
    template='plotly_white'
)
st.plotly_chart(fig4, use_container_width=True)


# --- 7. OUTPUT: Performance Summary ---
st.header('Step 7: Performance Summary')

# Calculate performance metrics
strategy_total_return_no_cost = (df['Strategy_Portfolio_Value_No_Cost'].iloc[-1] / df['Strategy_Portfolio_Value_No_Cost'].iloc[0] - 1) * 100
strategy_total_return_with_cost = (df['Strategy_Portfolio_Value'].iloc[-1] / df['Strategy_Portfolio_Value'].iloc[0] - 1) * 100
buy_hold_total_return = (df['Buy_Hold_Portfolio_Value'].iloc[-1] / df['Buy_Hold_Portfolio_Value'].iloc[0] - 1) * 100

days = (df.index[-1] - df.index[0]).days
years = days / 365.25

strategy_cagr_no_cost = ((df['Strategy_Portfolio_Value_No_Cost'].iloc[-1] / df['Strategy_Portfolio_Value_No_Cost'].iloc[0])**(1/years) - 1) * 100
strategy_cagr_with_cost = ((df['Strategy_Portfolio_Value'].iloc[-1] / df['Strategy_Portfolio_Value'].iloc[0])**(1/years) - 1) * 100
buy_hold_cagr = ((df['Buy_Hold_Portfolio_Value'].iloc[-1] / df['Buy_Hold_Portfolio_Value'].iloc[0])**(1/years) - 1) * 100

df['Strategy_Daily_Returns_Drawdown_No_Cost_Calc'] = df['Strategy_Portfolio_Value_No_Cost'].pct_change()
df['Strategy_Daily_Returns_Drawdown_With_Cost_Calc'] = df['Strategy_Portfolio_Value'].pct_change()
df['Strategy_Cumulative_Returns_No_Cost_Calc'] = (1 + df['Strategy_Daily_Returns_Drawdown_No_Cost_Calc']).cumprod()
df['Strategy_Cumulative_Returns_With_Cost_Calc'] = (1 + df['Strategy_Daily_Returns_Drawdown_With_Cost_Calc']).cumprod()
df['Strategy_Rolling_Max_Cumulative_No_Cost_Calc'] = df['Strategy_Cumulative_Returns_No_Cost_Calc'].cummax()
df['Strategy_Rolling_Max_Cumulative_With_Cost_Calc'] = df['Strategy_Cumulative_Returns_With_Cost_Calc'].cummax()

strategy_max_drawdown_no_cost = ((df['Strategy_Cumulative_Returns_No_Cost_Calc'] / df['Strategy_Rolling_Max_Cumulative_No_Cost_Calc']) - 1).min() * 100
strategy_max_drawdown_with_cost = ((df['Strategy_Cumulative_Returns_With_Cost_Calc'] / df['Strategy_Rolling_Max_Cumulative_With_Cost_Calc']) - 1).min() * 100
buy_hold_max_drawdown = ((df['Buy_Hold_Cumulative_Returns'] / df['Buy_Hold_Rolling_Max_Cumulative']) - 1).min() * 100

strategy_returns_no_cost = df['Strategy_Daily_Return_No_Cost'].dropna()
strategy_returns_with_cost = df['Strategy_Daily_Return'].dropna()
buy_hold_returns = df['Market_Daily_Return'].dropna()

strategy_annual_volatility_no_cost = strategy_returns_no_cost.std() * np.sqrt(252)
strategy_annual_volatility_with_cost = strategy_returns_with_cost.std() * np.sqrt(252)
buy_hold_annual_volatility = buy_hold_returns.std() * np.sqrt(252)

strategy_annual_return_no_cost = strategy_returns_no_cost.mean() * 252
strategy_annual_return_with_cost = strategy_returns_with_cost.mean() * 252
buy_hold_annual_return = buy_hold_returns.mean() * 252

strategy_sharpe_ratio_no_cost = strategy_annual_return_no_cost / strategy_annual_volatility_no_cost if strategy_annual_volatility_no_cost != 0 else np.nan
strategy_sharpe_ratio_with_cost = strategy_annual_return_with_cost / strategy_annual_volatility_with_cost if strategy_annual_volatility_with_cost != 0 else np.nan
buy_hold_sharpe_ratio = buy_hold_annual_return / buy_hold_annual_volatility if buy_hold_annual_volatility != 0 else np.nan

st.write("""
--- Performance Summary ---
""")
st.write(f"Initial Capital: INR {initial_capital:,}")
st.write(f"Backtest Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
st.write(f"Number of Trading Days: {len(df)}")

performance_data = {
    'Metric': ['Total Return (%)', 'CAGR (%)', 'Max Drawdown (%)', 'Sharpe Ratio'],
    'Strategy (No Cost)': [strategy_total_return_no_cost, strategy_cagr_no_cost, strategy_max_drawdown_no_cost, strategy_sharpe_ratio_no_cost],
    'Strategy (With Cost)': [strategy_total_return_with_cost, strategy_cagr_with_cost, strategy_max_drawdown_with_cost, strategy_sharpe_ratio_with_cost],
    'Buy & Hold': [buy_hold_total_return, buy_hold_cagr, buy_hold_max_drawdown, buy_hold_sharpe_ratio]
}
performance_df = pd.DataFrame(performance_data)
performance_df = performance_df.set_index('Metric')

st.dataframe(performance_df.style.format("{:.2f}"))
