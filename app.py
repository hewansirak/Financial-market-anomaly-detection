import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# Streamlit App Title
st.title("Investment Strategy Dashboard")

# Sidebar for User Input
st.sidebar.header("Select Assets")
assets = ["XAU/USD (Gold)", "CL1 (Crude Oil)", "DXY (Dollar Index)"]
selected_assets = st.sidebar.multiselect("Choose assets to view:", assets, default=assets)

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

# Display Charts for Selected Assets
for asset in selected_assets:
    ticker = asset_tickers[asset]
    st.subheader(f"{asset} - Real-Time Chart")

    # Fetch historical data
    data = fetch_data(ticker)

    if data is not None:
        # Create a Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=asset
        ))
        fig.update_layout(
            title=f"{asset} Price Movements",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display latest price
    ticker_data = yf.Ticker(ticker)
    current_price = ticker_data.history(period="1d")["Close"][-1]
    st.metric(label=f"{asset} Current Price", value=f"{current_price:.2f}")

# Footer
st.write("---")
st.write("Made with 💜 by Hewan")