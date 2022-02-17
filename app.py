import streamlit as st
from datetime import date
import yfinance as yf

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from pylab import rcParams

from plotly import graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("HP's STOCK PREDICTION APP")

st.sidebar.subheader("Enter Stock Details")


stocks = ('^AORD','^AXJO','AUDUSD=X','GC=F','CL=F','BTC-AUD','^CMC200')
selected_stock = st.sidebar.selectbox('Select the Stock',stocks)

n_years = st.sidebar.slider('Enter number of years',1,10)
y = n_years*365

st.subheader("The stock you selected")
st.write("Stock: ",selected_stock)
st.write("Years: ", n_years)


@st.cache
def data_load(ticker):
    data = yf.download(ticker,START,TODAY,progress=False)
    data.reset_index(inplace=True)
    return data


data = data_load(selected_stock)


st.subheader('Latest Stock Trend')
st.write(data.tail())


def plot_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name ="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name ="Stock Close"))
    fig.layout.update(title_text='Stock-Trend',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_data()


# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=y)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Data Forecast')

    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
