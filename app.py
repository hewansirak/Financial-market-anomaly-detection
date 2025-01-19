import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import joblib
import requests

# Load Isolation Forest Model
model_path = "isolation_forest_model.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'isolation_forest_model.pkl' is in the same directory.")
    st.stop()

# Streamlit App Title
st.title("Investment Strategy Dashboard")

# Sidebar for User Input
st.sidebar.header("Select Asset")
assets = ["XAU/USD (Gold)", "CL1 (Crude Oil)", "DXY (Dollar Index)"]
selected_asset = st.sidebar.selectbox("Choose an asset to view:", assets)

# Mapping asset names to Yahoo Finance tickers
asset_tickers = {
    "XAU/USD (Gold)": "GC=F",
    "CL1 (Crude Oil)": "CL=F",
    "DXY (Dollar Index)": "DX-Y.NYB"
}

# Fetch and Display Real-Time Data
def fetch_data(ticker):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1mo")
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Display Selected Asset's Real-Time Chart
ticker = asset_tickers[selected_asset]
st.subheader(f"{selected_asset} - Real-Time Chart")

# Fetch historical data
data = fetch_data(ticker)

if data is not None:
    # Create a Plotly candlestick chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=selected_asset
    ))
    fig.update_layout(
        title=f"{selected_asset} Price Movements",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display latest price
    try:
        current_price = data['Close'][-1]  # Access the latest price from the fetched data
        st.metric(label=f"{selected_asset} Current Price", value=f"{current_price:.2f}")
    except IndexError:
        st.error("Unable to fetch real-time price.")

# Custom Route for Anomaly Detection
st.write("### Anomaly Detection")

# User Input: Choose between real-time value or custom value
option = st.radio("Select Input Type:", ["Use Real-Time Value", "Enter Custom Value"])

if option == "Enter Custom Value":
    custom_value = st.number_input(f"Enter a custom value for {selected_asset}:", value=20.0, step=0.1)
else:
    if data is not None:
        custom_value = current_price  # Use the correctly accessed current_price
    else:
        st.warning("No real-time value available. Please enter a custom value.")
        custom_value = None

st.sidebar.header("Input Features")
VIX = st.sidebar.number_input("VIX", value=15.0, step=0.1)
DXY = st.sidebar.number_input("DXY", value=95.0, step=0.1)
GTDEM2Y = st.sidebar.number_input("GTDEM2Y", value=1.0, step=0.1)
EONIA = st.sidebar.number_input("EONIA", value=-0.3, step=0.1)
GTITL30YR = st.sidebar.number_input("GTITL30YR", value=2.0, step=0.1)
GTITL2YR = st.sidebar.number_input("GTITL2YR", value=0.5, step=0.1)
GTITL10YR = st.sidebar.number_input("GTITL10YR", value=1.5, step=0.1)
GTJPY30YR = st.sidebar.number_input("GTJPY30YR", value=0.6, step=0.1)
GTJPY2YR = st.sidebar.number_input("GTJPY2YR", value=0.1, step=0.1)

# Prediction Button
if st.button("Detect Anomaly"):
    # API Endpoint
    api_url = "http://127.0.0.1:8000/predict"
    
    # Input Data
    input_data = {
        "VIX": VIX,
        "DXY": DXY,
        "GTDEM2Y": GTDEM2Y,
        "EONIA": EONIA,
        "GTITL30YR": GTITL30YR,
        "GTITL2YR": GTITL2YR,
        "GTITL10YR": GTITL10YR,
        "GTJPY30YR": GTJPY30YR,
        "GTJPY2YR": GTJPY2YR
    }

    # Make API Request
    response = requests.post(api_url, json=input_data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Anomaly: {result['is_anomaly']}")
        st.write(f"Probability: {result['probability']:.2f}")
    else:
        st.error(f"Error: {response.json()['detail']}")
# Footer
st.write("---")
st.write("Made with 💜 by Hewan")

