import streamlit as st
from datetime import date

st.set_page_config(page_title="Welcome to IFRS13 Classification model", layout="wide")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/IFRS_logo.svg/1920px-IFRS_logo.svg.png", width=200)
st.sidebar.markdown("""
### IFRS13 Classification Workflow

Navigate through:
- **Model Inference**
- **Risk Factor Testing**
- **Rationale Explanation**

Use the sidebar to choose a module.
""")

st.markdown("""
<style>
.big-title {
    font-size: 2.8em;
    font-weight: 700;
    color: #0d6efd;
    margin-bottom: 0.2em;
    font-family: 'Segoe UI', sans-serif;
}
.subtitle {
    font-size: 1.3em;
    color: #333;
    margin-bottom: 1.5em;
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="big-title">🔍 Welcome to the IFRS13 Fair Value Classification App</div>
<div class="subtitle">A modular, explainable, and regulatory-friendly classification tool</div>
""", unsafe_allow_html=True)

st.info("Use the navigation on the left sidebar to begin.")

st.markdown("""
---
### 📌 Key Features:

- **Modular Design:** Split workflows for Model Inference, Observability, and Rationale
- **Regulatory Focus:** Built for IFRS13 compliance scenarios
- **Explainable AI:** Combines ML models and GPT rationale
- **Azure-native:** Integrates with Azure ML & OpenAI services

---
📅 Today's Date: **{}**
""".format(date.today().strftime("%d %b %Y")))
