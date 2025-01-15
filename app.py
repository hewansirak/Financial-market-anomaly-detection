import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Load pre-trained Isolation Forest model
@st.cache_resource
def load_model():
    with open("logistic_regression_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Simulated Data for Visualization
@st.cache_data
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "VIX": np.random.uniform(0, 50, 1000),
        "DXY": np.random.uniform(90, 110, 1000),
    })
    return data

# Load data
data = load_data()

# Streamlit App UI
st.title("AI-Driven Investment Strategy Bot")
st.write("This bot explains investment strategies based on market anomalies.")

# User Input
st.sidebar.header("Input Parameters")
vix_input = st.sidebar.slider("VIX Value (Volatility Index)", 0.0, 50.0, 25.0)
dxy_input = st.sidebar.slider("DXY Value (Dollar Index)", 90.0, 110.0, 100.0)

# Prediction
input_data = pd.DataFrame([[vix_input, dxy_input]], columns=["VIX", "DXY"])
prediction = model.predict(input_data)

# Display Prediction Results
st.subheader("Prediction Results")
if prediction[0] == -1:
    st.write("📉 **Anomaly Detected**")
    st.warning("Consider safe investments like bonds, gold, or cash equivalents.")
else:
    st.write("📈 **No Anomaly Detected**")
    st.success("You can pursue higher-risk strategies or diversify equity investments.")

# Visualizations
st.subheader("Visualizations")

# Scatter Plot of Data
st.write("### Market Data Distribution")
fig, ax = plt.subplots()
sns.scatterplot(
    x="VIX", y="DXY", data=data, alpha=0.7, edgecolor=None, label="Data Points"
)
plt.scatter(
    vix_input, dxy_input, color="red", label="Your Input", s=100, edgecolor="black"
)
plt.xlabel("VIX")
plt.ylabel("DXY")
plt.title("VIX vs DXY")
plt.legend()
st.pyplot(fig)

# Model Decision Regions
st.write("### Model Decision Regions")
xx, yy = np.meshgrid(
    np.linspace(data["VIX"].min(), data["VIX"].max(), 100),
    np.linspace(data["DXY"].min(), data["DXY"].max(), 100),
)
grid_data = np.c_[xx.ravel(), yy.ravel()]
decision = model.decision_function(grid_data).reshape(xx.shape)

fig, ax = plt.subplots()
contour = plt.contourf(
    xx, yy, decision, levels=np.linspace(decision.min(), decision.max(), 50), cmap="coolwarm", alpha=0.8
)
cbar = plt.colorbar(contour)
cbar.set_label("Anomaly Score")
plt.scatter(data["VIX"], data["DXY"], c="white", s=10, alpha=0.5, label="Data Points")
plt.scatter(
    vix_input, dxy_input, color="red", label="Your Input", s=100, edgecolor="black"
)
plt.xlabel("VIX")
plt.ylabel("DXY")
plt.title("Isolation Forest Decision Regions")
plt.legend()
st.pyplot(fig)

# Educational Section
st.sidebar.header("Learn More")
if st.sidebar.button("What is VIX?"):
    st.sidebar.info(
        "The Volatility Index (VIX) measures market expectations for volatility. "
        "High VIX values often indicate uncertainty or instability."
    )
if st.sidebar.button("What is DXY?"):
    st.sidebar.info(
        "The US Dollar Index (DXY) measures the strength of the US Dollar relative "
        "to a basket of foreign currencies. High DXY values can signal a strong dollar."
    )

# Footer
st.write("---")
st.write("Made with 💜 by Hewan")
