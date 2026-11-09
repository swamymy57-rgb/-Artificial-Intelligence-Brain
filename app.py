import streamlit as st
from my_ai_brain.frontend import render_app

st.set_page_config(
    page_title="Personal AI Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

if __name__ == "__main__":
    render_app()
  
