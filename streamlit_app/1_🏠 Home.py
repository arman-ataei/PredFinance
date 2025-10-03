import streamlit as st
import os, sys
# st.title("🤖 Train/Compare Models")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

def show():
    st.title("🏠 Home")
    st.write("Welcome to the BitCoin Price Predictor Streamlit app!")
    st.markdown("""
    - **EDA**: In this page it i
    - **Models**: Train and evaluate ML models  
    - **Inference**: Run predictions on new data  
    """)


st.title("🏠 Home")
st.html("<h2>Welcome to the BitCoin Price Predictor!</h2>")
st.markdown("""
TThis application is released under the MIT License, provided “as is” without any warranty, and is intended solely for educational purposes. It has been built using Streamlit.
The app is organized into two main sections:
 
- **Data Visualization**: Explore the dataset interactively, along with extracted features and indicators, using dynamic Plotly charts.
- **Models**: Train new models or load pre-trained ones to evaluate their performance and compare results.
""")