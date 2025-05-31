import streamlit as st
from workflow_styles import get_workflow_css

st.set_page_config(page_title="Dual Lane with Uniform Colors", layout="wide")

st.markdown(get_workflow_css(), unsafe_allow_html=True)

st.markdown(
"""

<!-- Lane 1: Model Inference -->
<div class="lane">
    <div class="header model">Model Inference</div>
    <div class="horizontal-body">
        <div class="step-model" style="animation-delay: 0s;">Trade Details</div>
        <div class="arrow-model">▶</div>
        <div class="step-model" style="animation-delay: 0s;">Model Input</div>
        <div class="arrow-model">▶</div>
        <div class="step-model" style="animation-delay: 0.2s;">Model Inference</div>
        <div class="arrow-model">▶</div>
        <div class="step-model" style="animation-delay: 0.4s;">Level Predicted by Model</div>
    </div>
</div>

<!-- Lane 2: Risk Factor Observability -->
<div class="lane">
<div class="header stress">Risk Factor Observability Inference</div>

<!-- Horizontal stepper -->
<div class="horizontal-body">
    <div class="step-stress" style="animation-delay: 0s;">Trade Details</div>
    <div class="arrow-stress">▶</div>
    <div class="step-stress" style="animation-delay: 0.2s;">Source Risk Factors</div>
    <div class="arrow-stress">▶</div>
    <div class="step-stress" style="animation-delay: 0.4s;">Observability Test</div>
    <div class="arrow-stress">▶</div>
    <div class="step-stress" style="animation-delay: 0.6s;">Level Predicted by Risk Testing</div>
</div>
</div>

""",
unsafe_allow_html=True
)
