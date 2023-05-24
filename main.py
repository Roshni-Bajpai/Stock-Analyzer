import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from datetime import date, datetime,time
from datetime import timedelta
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

stocks_list = [
    'ONGC.NS', 'UPL.NS', 'ITC.NS', 'SUNPHARMA.NS', 'IOC.NS', 'JSWSTEEL.NS',
    'SBIN.NS', 'SHREECEM.NS', 'HINDUNILVR.NS', 'NTPC.NS', 'HINDALCO.NS',
    'LT.NS', 'BAJFINANCE.NS', 'DIVISLAB.NS', 'TATACONSUM.NS', 'HDFCLIFE.NS',
    'M&M.NS', 'INFY.NS', 'GRASIM.NS', 'WIPRO.NS', 'COALINDIA.NS',
    'BRITANNIA.NS', 'INDUSINDBK.NS', 'BHARTIARTL.NS', 'SBILIFE.NS',
    'ICICIBANK.NS', 'TATASTEEL.NS', 'RELIANCE.NS', 'HCLTECH.NS',
    'BAJAJ-AUTO.NS', 'BPCL.NS', 'TCS.NS', 'NESTLEIND.NS', 'ADANIPORTS.NS',
    'AXISBANK.NS', 'ULTRACEMCO.NS', 'CIPLA.NS', 'TITAN.NS', 'HEROMOTOCO.NS',
    'KOTAKBANK.NS', 'BAJAJFINSV.NS', 'POWERGRID.NS', 'ASIANPAINT.NS',
    'EICHERMOT.NS', 'TATAMOTORS.NS', 'DRREDDY.NS', 'HDFCBANK.NS', 'HDFC.NS',
    'MARUTI'
]

st.set_page_config(layout="wide")
st.title("Stock Analysis On NIFTY50")
st.markdown(
    "<h2 style='text-align: center; color: white;'>Stock Analysis On NIFTY50</h2>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: white;'>Detailed Analysis and Visualization for Top 50 NIFTY Stocks</p>",
    unsafe_allow_html=True)

options = st.selectbox("Select a Stock", stocks_list)
end = datetime.today().date()
start = end - timedelta(days=30)


def stock_fn(options):
    df = yf.download(tickers=options, period='5d', interval='1m')
    return df


def stock_fn1(options, start, end):
    df = yf.download(options, end=end, start=start)
    df = df.reset_index()
    df["MarktCap"] = df["Open"] * df["Volume"]
    df["MA50"] = df["Open"].rolling(50).mean()
    df["MA200"] = df["Open"].rolling(200).mean()
    df["returns"] = ((df["Close"] / df["Close"].shift(1)) - 1) * 100
    return df


stocks_df = stock_fn(options)
stock = stock_fn1(options, start, end)
print(stock.head())
closing_price = round(stocks_df['Close'].iloc[-1], 2)
return_percent = round(stock['returns'].iloc[-1], 2)
return_value = round((stocks_df["Close"].iloc[-1] - stocks_df["Close"].iloc[-2]), 2)

if return_value >= 0:
    rotation = 0
    clr = "#42FF00"
    height = "50px"
else:
    rotation = 180
    clr = "#FF0000"
    height = "65px"
tab = f"""
<html>

<head>
    <link href="https://fonts.googleapis.com/css?family=Montserrat&display=swap" rel="stylesheet" />
    <style></style>
</head>

<body>
    <div style="        
    width: 400px;
    height: 96px;
    background-repeat: no-repeat;
    background-position: center center;
    background-size: cover;
    opacity: 1;
    position: relative;
    top: 0px;
    left: 0px;
    overflow: hidden;">
        <div style="        
        width: 250px;
        height: 96px;
        background: rgba(38, 39, 48, 1);
        opacity: 1;
        position: relative;
        top: 0px;
        left: 0px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        border-bottom-left-radius: 6px;
        border-bottom-right-radius: 6px;
        overflow: hidden;">
        </div>
        <div style="        
        position: absolute;
        top:{height};
        left: 15px;
        width: 0;
        height: 0;
        transform: rotate({rotation}deg);
        border: solid 15px;
        border-color: transparent transparent {clr} transparent;"></div>
        <span style="        
            width: 190px;
            color: white;
            position: absolute;
            top: 2px;
            left: 15px;
            font-family: Montserrat;
            font-weight: SemiBold;
            font-size: 38px;
            opacity: 1;
            text-align: center;">{closing_price}</span>
        <span style="        
            width: 130px;
            color:{clr};
            position: absolute;
            top: 51px;
            left: 55px;
            font-family: Montserrat;
            font-weight: Medium;
            font-size: 26px;
            opacity: 1;
            text-align: left;"> {return_value}</span>
        <span style="        
            width: 130px;
            color:{clr};
            position: absolute;
            top: 51px;
            left:130px;
            font-family: Montserrat;
            font-weight: Medium;
            font-size: 24px;
            opacity: 1;
            text-align: left;">({return_percent}%)</span>
    </div>
</body>

</html><br>"""

with open("tabcss.html", "r") as f:
    tabcss = f.read()
tab_bar = tab.replace("<style></style>", tabcss)
st.markdown(tab_bar, unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=stock["Date"],
        open=stock["Open"],
        high=stock["High"],
        low=stock["Low"],
        close=stock["Close"],
    ))

fig.update_layout(
    hovermode="x unified",
    autosize=False,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    title={
        "text":
            f"<span style='font-size: 20px;'>{options}</span><span style='font-size: 24px;color:#9FE6A0'> HIGH</span><span style='font-size: 24px;color:white'> vs</span><span style='font-size: 24px;color:#F55C47'> LOW</span>",
        "x": 0.5,
        "xanchor": "center",
        'yanchor': 'top'
    },
    margin=dict(
        l=10,  # left
        r=10,  # right
        t=60,  # top
        b=10,  # bottom
    ),
    paper_bgcolor="rgb(38,39,48)",
    plot_bgcolor="rgb(38,39,48)",
)
st.plotly_chart(fig, use_container_width=True)

#################################
fig = go.Figure()

# Add traces
fig.add_trace(go.Line(x=stock["Date"], y=stock["Open"], name=f"{options}"))

fig.update_layout(
    hovermode="x unified",
    template="plotly_dark",
    title={
        "text":
            f"<span style='font-size: 20px;'>{options}</span> <span style='font-size: 24px;color:#edf5ee'> Opening Price</span>",
        "x": 0.5,
        "xanchor": "center",
    },
    margin=dict(
        l=10,  # left
        r=10,  # right
        t=60,  # top
        b=10,  # bottom
    ),
    paper_bgcolor="rgb(38,39,48)",
    plot_bgcolor="rgb(38,39,48)",
)
st.plotly_chart(fig, use_container_width=True)

fig = go.Figure()

# Add traces
fig.add_trace(go.Line(x=stock["Date"], y=stock["Volume"], name="TCS"))

fig.update_layout(
    hovermode="x unified",
    template="plotly_dark",
    title={
        "text":
            f"<span style='font-size: 20px;'>{options}</span> <span style='font-size: 24px;color:#edf5ee'> Volume Traded</span>",
        "x": 0.5,
        "xanchor": "center",
    },
    margin=dict(
        l=10,  # left
        r=10,  # right
        t=60,  # top
        b=10,  # bottom
    ),
    paper_bgcolor="rgb(38,39,48)",
    plot_bgcolor="rgb(38,39,48)",
)
st.plotly_chart(fig, use_container_width=True)

########################################33333
fig = go.Figure()
# Add traces
fig.add_trace(go.Line(x=stock["Date"], y=stock["MarktCap"], name=f"{options}"))

fig.update_layout(
    hovermode="x unified",
    template="plotly_dark",
    title={
        "text":
            f"<span style='font-size: 20px;'>{options}</span> <span style='font-size: 24px;color:#edf5ee'> Market Cap</span>",
        "x": 0.5,
        "xanchor": "center",
    },
    margin=dict(
        l=10,  # left
        r=10,  # right
        t=60,  # top
        b=10,  # bottom
    ),
    paper_bgcolor="rgb(38,39,48)",
    plot_bgcolor="rgb(38,39,48)",
)
st.plotly_chart(fig, use_container_width=True)

###########################################################

fig = go.Figure()
# Add traces
fig.add_trace(go.Line(x=stock["Date"], y=stock["Open"], name=f"{options}"))
fig.add_trace(go.Line(x=stock["Date"], y=stock["MA50"], name="Moving Avg 50"))
fig.add_trace(go.Line(x=stock["Date"], y=stock["MA200"],
                      name="Moving Avg 200"))

# fig.update_layout(
#     hovermode="x unified",
#     template="plotly_dark",
#     title={
#         "text":
#         f"<span style='font-size: 20px;'>{options}</span> <span style='font-size: 24px;'> Opening Trend with Moving Avg</span>",
#         "x": 0.5,
#         "xanchor": "center",
#     },
#     margin=dict(
#         l=10,  # left
#         r=10,  # right
#         t=60,  # top
#         b=10,  # bottom
#     ),
#     paper_bgcolor="rgb(38,39,48)",
#     plot_bgcolor="rgb(38,39,48)",
# )
# st.plotly_chart(fig, use_container_width=True)

######################################3333
# fig = go.Figure()
#
# # Add traces
# fig.add_trace(go.Histogram(
#     x=stock["returns"],
#     name=f"{options} Returns",
# ))
#
# fig.update_layout(
#     xaxis_title="Share % Return",
#     hovermode="x unified",
#     barmode="overlay",
#     template="plotly_dark",
#     title={
#         "text":
#         f"<span style='font-size: 20px;'>{options}</span> <span style='font-size: 24px;'> Volatility</span>",
#         "x": 0.5,
#         "xanchor": "center",
#     },
#     margin=dict(
#         l=10,  # left
#         r=10,  # right
#         t=60,  # top
#         b=10,  # bottom
#     ),
#     paper_bgcolor="rgb(38,39,48)",
#     plot_bgcolor="rgb(38,39,48)",
# )
# st.plotly_chart(fig, use_container_width=True)

# from statsmodels.tsa.stattools import adfuller
# # creating a function for values
# def adf_test(dataset):
#     dftest = adfuller(dataset)
#     print("1. ADF : ", dftest[0])
#     print("2. P-Value : ", dftest[1])
#     print("3. Num Of Lags : ", dftest[2])
#     print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
#     print("5. Critical Values :")
#     for key, val in dftest[4].items():
#         print("\t", key, ": ", val)
#
# print(adf_test(stocks_df))
stepwise_fit = auto_arima(stocks_df['Close'], suppress_warnings=True)

train = stocks_df.iloc[:-30]
test = stocks_df.iloc[-30:]

model = sm.tsa.arima.ARIMA(stocks_df['Close'], order=(1, 0, 5))
model = model.fit()

start = len(train)
end = len(train) + len(test) - 1
pred = model.predict(start=start, end=end, typ='levels').rename('ARIMA predictions')
# pred.index=index_future_dates
fig1 = pred.plot(legend=True)
fig = test['Close'].plot(legend=True)

# now = datetime.now().time()
# current_time = now.strftime("%H:%M")
# today = datetime.today()
# x=time(15,30)
# clo_time = datetime.combine(today,x)
# closing_time = clo_time.strftime("%H:%M")
# time_diff = closing_time - current_time
# minutes_diff = int(time_diff.total_seconds() / 60)



model2 = sm.tsa.arima.ARIMA(stocks_df['Close'], order=(1, 0, 5))
model2 = model2.fit()
pred1 = stepwise_fit.predict(start=len(stocks_df), end=len(stocks_df), typ='levels').rename('ARIMA Predictions')
mean_df = pred1.mean()
rmse = sqrt(mean_squared_error(pred, test['Close']))

st.header("Closing Price Predictor")
label = "Closing Price of the current day"
value = mean_df
from streamlit_card import card

st.metric(label, value, delta=None, delta_color="normal", help=None, label_visibility="visible")
