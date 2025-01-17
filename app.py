import streamlit as st
import pandas as pd
import mplfinance as mpf
from datetime import datetime

# Streamlit Configuration
st.set_page_config(layout="wide", page_title="Candlestick Chart App")

# Load and preprocess the dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv("data.csv")
    df["Data"] = pd.to_datetime(df["Data"], format="%m/%d/%Y")  # Convert 'Data' column to datetime
    df.set_index("Data", inplace=True)  # Set 'Data' as index
    return df

data = load_dataset()

# Sidebar Inputs
st.sidebar.header("Candlestick Chart Configuration")

# Date Selection
start_date = st.sidebar.date_input("Start Date", value=datetime(2000, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2000, 12, 31))

# Plot Style
plot_styles = ["classic", "charles", "yahoo", "mike", "nightclouds", "sas"]
plot_style = st.sidebar.selectbox("Select Chart Style", plot_styles, index=3)

# Volume Display Toggle
show_volume = st.sidebar.checkbox("Show Volume", value=False)

# Filter data based on user inputs
filtered_data = data.loc[start_date:end_date]

# Mock Open, High, Low, Close columns (since they are missing)
if not filtered_data.empty:
    filtered_data["Open"] = filtered_data["XAU BGNL"] * 0.98  # Mock Open as 98% of Close
    filtered_data["High"] = filtered_data["XAU BGNL"] * 1.02  # Mock High as 102% of Close
    filtered_data["Low"] = filtered_data["XAU BGNL"] * 0.97   # Mock Low as 97% of Close
    filtered_data["Close"] = filtered_data["XAU BGNL"]        # Use 'XAU BGNL' as Close

    # Columns required for candlestick
    ohlc = filtered_data[["Open", "High", "Low", "Close"]]

    # Plot Candlestick Chart using mplfinance
    st.title(":green[Candlestick] :red[Chart]")
    st.subheader("Custom Dataset Candlestick Chart")

    fig, axlist = mpf.plot(
        ohlc,
        type="candle",
        style=plot_style,
        volume=show_volume,
        title="Candlestick Chart",
        ylabel="Price",
        ylabel_lower="Volume",
        returnfig=True
    )

    # Display the chart in Streamlit
    st.pyplot(fig)
else:
    st.error("No data available for the selected date range.")

# Display Raw Data
if st.checkbox("Show Raw Data"):
    st.write(filtered_data)
