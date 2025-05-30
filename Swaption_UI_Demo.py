import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import date
from Observability_Stress_Module import run_full_observability_stress_test, simulate_greeks

# --- Initialize Session State ---
if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = 0

# --- OIS Curve Mapping ---
ois_curve_map = {
    "USD": "USD.OIS",
    "EUR": "EUR.OIS",
    "GBP": "GBP.OIS",
    "JPY": "JPY.OIS"
}

# --- Load Observability Grids ---
ir_grid = pd.read_csv("ir_delta_observability_grid.csv")
ir_grid.columns = ir_grid.columns.str.strip()
ir_grid["Observable Tenor (Years)"] = pd.to_numeric(ir_grid["Observable Tenor (Years)"], errors="coerce")

vol_grid = pd.read_csv("volatility_observability_grid.csv")
vol_grid.columns = vol_grid.columns.str.strip()

# --- Page Config ---
st.set_page_config(page_title="Swaption - IFRS13 Classification", layout="wide")
st.title("Swaption - IFRS13 Observability Classification Orchestrator")

# --- Stepper UI ---
step_labels = ["1. Model Prediction", "2. Simulate Greeks", "3. Stress Test", "4. Explain Rationale"]
step_html = '<div class="stepper">'
for i, label in enumerate(step_labels):
    if st.session_state.workflow_step == i + 1:
        step_html += f'<div class="step"><span class="active">{label}</span></div>'
    else:
        step_html += f'<div class="step"><span>{label}</span></div>'
step_html += '</div>'

st.markdown(f"""
<style>
.stepper {{
  display: flex;
  justify-content: space-between;
  margin-bottom: 1em;
  font-weight: bold;
  padding: 10px;
}}
.step {{
  width: 100%;
  text-align: center;
  position: relative;
}}
.step::before {{
  content: "";
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 2px;
  background: #444;
  z-index: 0;
}}
.step span {{
  background: #222;
  color: #ddd;
  padding: 0.5em 1em;
  border-radius: 20px;
  position: relative;
  z-index: 1;
  display: inline-block;
  border: 1px solid #888;
}}
.step span.active {{
  background: #0d6efd;
  color: white;
  border: none;
}}
</style>
{step_html}
""", unsafe_allow_html=True)

# --- Custom Progress Bar ---
def render_custom_progress(percent):
    st.markdown(f"""
    <div style="background-color: #e0e0e0; border-radius: 20px; height: 24px; width: 100%; margin-top: 10px; margin-bottom: 25px;">
        <div style="background-color: #0d6efd; width: {percent}%; height: 100%; border-radius: 20px; text-align: center; color: white; line-height: 24px;">
            {percent}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("Swaption Trade Inputs")
product_type = st.sidebar.selectbox("Product Type", ["IR Swaption", "Bond", "CapFloor", "IRSwap"], index=0)
notional = st.sidebar.number_input("Notional", min_value=1_000_000, step=1_000_000, value=10_000_000)
currency = st.sidebar.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
strike = st.sidebar.slider("Strike (%)", 0.0, 10.0, 2.5, 0.1)

expiry_tenor = st.sidebar.selectbox("Expiry Tenor (Y)", [2, 3, 5])
expiry_date = date.today().replace(year=date.today().year + expiry_tenor)
st.sidebar.markdown(f"📅 **Expiry Date**: {expiry_date.strftime('%d-%b-%Y')}")

maturity_tenor = st.sidebar.selectbox("Maturity Tenor (Y)", [5, 10, 15, 20, 30])
maturity_date = date.today().replace(year=date.today().year + maturity_tenor)
st.sidebar.markdown(f"📅 **Maturity Date**: {maturity_date.strftime('%d-%b-%Y')}")

# --- Trade Dictionary ---
trade = {
    "product_type": product_type,
    "currency": currency,
    "expiry_tenor": expiry_tenor,
    "maturity_tenor": maturity_tenor,
    "strike": strike,
    "notional": notional
}

# --- Model Logic ---
def mock_model_prediction(trade):
    if trade["expiry_tenor"] < 5 and trade["maturity_tenor"] < 15 and trade["strike"] < 3.0:
        return "Level 2"
    return "Level 3"

# --- Workflow Execution ---
if st.button("Run Full IFRS 13 Classification Workflow"):

    # Step 1 - Prediction
    st.session_state.workflow_step = 1
    st.subheader("1. Machine Learning Model Prediction")
    # Trade JSON
    st.markdown("### 📦 Trade JSON (Input to ML Model)")
    st.code(json.dumps(trade, indent=2), language='json')
    model_pred = mock_model_prediction(trade)
    st.session_state["ifrs13_level"] = model_pred
    st.success(f"Predicted IFRS13 Level: {model_pred}")
    render_custom_progress(25)
    # Display in sidebar immediately after prediction
    level_html = f"""<div style='
        background-color:#d4edda;
        color:#155724;
        padding:10px;
        border-left:5px solid #28a745;
        border-radius:5px;
        margin-top:20px;
        font-weight:bold'>
        IFRS13 Level:<br>{model_pred}
    </div>"""
    st.sidebar.markdown(level_html, unsafe_allow_html=True)

   

    # Step 2 - Greeks Simulation
    st.session_state.workflow_step = 2
    st.subheader("2. Simulating Risk Factors")
    greeks = simulate_greeks(trade)
    st.json(greeks)
    render_custom_progress(50)

    # Step 3 - Stress Test
    st.session_state.workflow_step = 3
    st.subheader("3. Observability-Based Stress Test")
    stressed, report, messages = run_full_observability_stress_test(trade, greeks, ir_grid, vol_grid, ois_curve_map)

    report_df = pd.DataFrame(report).T
    report_df.index.name = "Risk Factor"
    st.markdown("### 🔍 Observability Report")
    st.dataframe(report_df)

    st.markdown("### 📊 Summary Metrics")
    st.metric("Total Trade PV", f"{stressed['Total Trade PV']:,.2f}")
    st.metric("Total Stress PV from Unobservable Risks", f"{stressed['Total Stress PV']:,.2f}")

    if stressed["Total Stress PV"] > 0.1 * stressed["Total Trade PV"]:
        st.error("🔴 Stress impact exceeds 10% of Trade PV → Level 3")
    else:
        st.success("🟢 Stress impact within 10% of Trade PV → Level 2")

    render_custom_progress(75)
    st.session_state.workflow_step = 4

# --- Final Step - Rationale (Always visible) ---
st.subheader("4. Explain Rationale")
st.info("🔍 Rationale explanation will be generated via Azure AI Foundry (gpt-4o API integration coming soon).")
if st.session_state.workflow_step == 4:
    render_custom_progress(100)
