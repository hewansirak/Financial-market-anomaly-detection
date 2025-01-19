import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import joblib
import requests
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

model_path = "isolation_forest_model.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'isolation_forest_model.pkl' is in the same directory.")
    st.stop()

st.title("Investment Strategy Dashboard")

st.sidebar.header("Select Asset")
assets = ["XAU/USD (Gold)", "CL1 (Crude Oil)", "DXY (Dollar Index)"]
selected_asset = st.sidebar.selectbox("Choose an asset to view:", assets)

asset_tickers = {
    "XAU/USD (Gold)": "GC=F",
    "CL1 (Crude Oil)": "CL=F",
    "DXY (Dollar Index)": "DX-Y.NYB"
}

def fetch_data(ticker):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1mo")
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

ticker = asset_tickers[selected_asset]
st.subheader(f"{selected_asset} - Real-Time Chart")

data = fetch_data(ticker)

if data is not None:
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

    try:
        current_price = data['Close'][-1]
        st.metric(label=f"{selected_asset} Current Price", value=f"{current_price:.2f}")
    except IndexError:
        st.error("Unable to fetch real-time price.")

st.write("### Anomaly Detection")

option = st.radio("Select Input Type:", ["Use Real-Time Value", "Enter Custom Value"])

if option == "Enter Custom Value":
    custom_value = st.number_input(f"Enter a custom value for {selected_asset}:", value=20.0, step=0.1)
else:
    if data is not None:
        custom_value = current_price 
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

api_key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def generate_explanation(data, selected_asset, current_price, is_anomaly, probability):
    """
    Generates an investment strategy explanation based on model predictions.

    Args:
        data: Historical data for the selected asset.
        selected_asset: Name of the selected asset (e.g., "XAU/USD (Gold)").
        current_price: Current price of the selected asset.
        is_anomaly: Boolean indicating whether the model predicts an anomaly.
        probability: Probability of the anomaly prediction.

    Returns:
        A string containing the generated explanation.
    """

    prompt = f"""
    You are an expert quantitative trader. 
    You are tasked to propose and advice a data-driven investment strategy for {selected_asset} given the following:
    - Current price: {current_price} 
    - Model prediction: {'Anomaly' if is_anomaly else 'Not an Anomaly'}
    - Anomaly probability: {probability}

    Focus on minimizing losses or maximizing returns. Explain the strategy to end users in an accessible and actionable manner. 

    In your response:
        - Make digits with only 2 number after a decimal point
        - Make it essay like in paragraphs
        - Start with a concise statement of the investment strategy.
        - Explain the rationale behind the strategy, referencing market conditions or observed trends.
        - Discuss potential market drivers or factors contributing to the model's prediction.
        - Avoid mentioning the model itself or its inner workings.
        - Use quantitative trader terminology where appropriate.
        - Keep your explanation concise and to the point.

    Let's think step by step about this. Verify step by step.
    """

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile"  
    )

    return response.choices[0].message.content

if st.button("Detect Anomaly"):
    api_url = "http://127.0.0.1:8000/predict"
    
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

    response = requests.post(api_url, json=input_data)
    if response.status_code == 200:
        result = response.json()
        is_anomaly = result['is_anomaly']
        probability = result['probability']

        explanation = generate_explanation(data, selected_asset, current_price, is_anomaly, probability) 

        st.success(f"Anomaly: {result['is_anomaly']}")
        st.write(f"Probability: {result['probability']:.2f}")
        st.write("**Investment Strategy:**")
        st.write(explanation) 

    else:
        st.error(f"Error: {response.json()['detail']}")


st.write("---")
st.write("Made with 💜 by Hewan")

