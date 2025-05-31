import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import date
import openai
from Observability_Stress_Module import run_full_observability_stress_test, simulate_greeks, ir_delta_stress_test, vol_risk_stress_test, generate_trade_pv_and_risk_pvs, ois_curve_map
from workflow_styles import get_workflow_css, get_workflow_html_ml, get_workflow_html_rf, get_workflow_html_rat

# --- Initialize Session State ---
for key in ["ml_step", "rf_step", "rat_step", "show_ml_workflow", "show_rf_workflow", "show_rat_workflow"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if "step" in key else False

# --- Load Observability Grids ---
ir_grid = pd.read_csv("ir_delta_observability_grid.csv")
ir_grid.columns = ir_grid.columns.str.strip()
ir_grid["Observable Tenor (Years)"] = pd.to_numeric(ir_grid["Observable Tenor (Years)"], errors="coerce")
vol_grid = pd.read_csv("volatility_observability_grid.csv")
vol_grid.columns = vol_grid.columns.str.strip()

# --- Page Config ---
st.set_page_config(page_title="Swaption - IFRS13 Classification", layout="wide")
st.title("Swaption - IFRS13 Observability Classification")

# --- Workflow CSS ---
st.markdown(get_workflow_css(), unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("Swaption Trade Inputs")
product_type = st.sidebar.selectbox("Product Type", ["IR Swaption", "Bond", "CapFloor", "IRSwap"], index=0)
notional = st.sidebar.number_input("Notional", min_value=1_000_000, step=1_000_000, value=10_000_000)
currency = st.sidebar.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
strike = st.sidebar.slider("Strike (%)", 0.0, 10.0, 2.5, 0.1)
expiry_tenor = st.sidebar.selectbox("Expiry Tenor (Y)", [2, 3, 5])
expiry_date = date.today().replace(year=date.today().year + expiry_tenor)
st.sidebar.markdown(f"\U0001F4C5 **Expiry Date**: {expiry_date.strftime('%d-%b-%Y')}")
maturity_tenor = st.sidebar.selectbox("Maturity Tenor (Y)", [5, 10, 15, 20, 30])
maturity_date = date.today().replace(year=date.today().year + maturity_tenor)
st.sidebar.markdown(f"\U0001F4C5 **Maturity Date**: {maturity_date.strftime('%d-%b-%Y')}")

# --- Trade Dictionary ---
trade = {
    "product_type": product_type,
    "currency": currency,
    "expiry_tenor": expiry_tenor,
    "maturity_tenor": maturity_tenor,
    "strike": strike,
    "notional": notional
}

# --- Model Prediction Mock ---
def mock_model_prediction(trade):
    if trade["expiry_tenor"] < 5 and trade["maturity_tenor"] < 15 and trade["strike"] < 3.0:
        return "Level 2"
    return "Level 3"

# --- Section: Machine Learning Model Prediction ---
st.subheader("1. Machine Learning Model Prediction")
st.markdown(get_workflow_html_ml(st.session_state.ml_step), unsafe_allow_html=True)

with st.expander("Step 1: Prepare ML Input", expanded=st.session_state.ml_step >= 1):
    if st.button("🔁 Run Step 1: Prepare ML Input") or st.session_state.ml_step >= 1:
        st.session_state.ml_step = max(st.session_state.ml_step, 1)
        st.code(json.dumps(trade, indent=2), language='json')
        st.success("✅ ML Input ready")

with st.expander("Step 2: Featurization", expanded=st.session_state.ml_step >= 2):
    if st.button("🧮 Run Step 2: Featurization") or st.session_state.ml_step >= 2:
        st.session_state.ml_step = max(st.session_state.ml_step, 2)
        st.info("Features extracted successfully (mock)")

with st.expander("Step 3: ML Model Inference", expanded=st.session_state.ml_step >= 3):
    if st.button("🤖 Run Step 3: Model Inference") or st.session_state.ml_step >= 3:
        if st.session_state.ml_step < 3:
            model_pred = mock_model_prediction(trade)
            st.session_state["ifrs13_level"] = model_pred
            st.session_state["model_pred"] = model_pred
        st.session_state.ml_step = max(st.session_state.ml_step, 3)
        st.success("ML pipeline executed")
        st.success(f"Predicted IFRS13 Level: {st.session_state['model_pred']}")
        level_html = f"""<div style='background-color:#d4edda;color:#155724;padding:10px;border-left:5px solid #28a745;border-radius:5px;margin-top:20px;font-weight:bold'>IFRS13 Level:<br>{st.session_state['model_pred']}</div>"""
        st.sidebar.markdown(level_html, unsafe_allow_html=True)

# --- Section: Risk Factor-based Inference ---
st.subheader("2. Risk Factor-based Inference")
st.markdown(get_workflow_html_rf(st.session_state.rf_step), unsafe_allow_html=True)

with st.expander("Step 1: Source Risk Factors", expanded=st.session_state.rf_step >= 1):
    if st.button("📊 Run Step 1") or st.session_state.rf_step >= 1:
        if st.session_state.rf_step < 1:
            st.session_state.rf_step = 1
            st.session_state.greeks = simulate_greeks(trade)
        st.success("✅ Greeks simulated successfully")

with st.expander("Step 2: Run IR Delta Observability Test", expanded=st.session_state.rf_step >= 2):
    if st.button("📈 Run Step 2") or st.session_state.rf_step >= 2:
        if st.session_state.rf_step < 2:
            st.session_state.rf_step = 2
            greeks = st.session_state.greeks
            st.session_state.ir_result = ir_delta_stress_test(trade, greeks)

with st.expander("Step 3: Run Volatility Observability Test", expanded=st.session_state.rf_step >= 3):
    if st.button("📉 Run Step 3") or st.session_state.rf_step >= 3:
        if st.session_state.rf_step < 3:
            st.session_state.rf_step = 3
            greeks = st.session_state.greeks
            st.session_state.vol_result = vol_risk_stress_test(trade, greeks)

with st.expander("Step 4: Assess Observability of Total PV", expanded=st.session_state.rf_step >= 4):
    if st.button("🧮 Run Step 4") or st.session_state.rf_step >= 4:
        if st.session_state.rf_step < 4:
            st.session_state.rf_step = 4
        ir_stressed, ir_report, ir_stress_pv, ir_msgs = st.session_state.ir_result
        vol_stressed, vol_report, vol_stress_pv, vol_msgs = st.session_state.vol_result

        st.markdown("### IR Delta Observability Report")
        st.dataframe(pd.DataFrame(ir_report).T)
        for msg in ir_msgs:
            st.warning(msg)

        st.markdown("### Volatility Observability Report")
        st.dataframe(pd.DataFrame(vol_report).T)
        for msg in vol_msgs:
            st.warning(msg)

        trade_pv, _ = generate_trade_pv_and_risk_pvs(st.session_state.greeks)
        total_stress_pv = ir_stress_pv + vol_stress_pv
        st.session_state["summary_ir"] = ir_msgs
        st.session_state["summary_vol"] = vol_msgs
        final_level = "Level 3" if total_stress_pv > 0.1 * trade_pv else "Level 2"
        st.metric("Total Stress PV", total_stress_pv)
        if final_level == "Level 3":
            st.error("🔴 Unobservable stress exceeds threshold → Level 3")
        else:
            st.success("🟢 Within threshold → Level 2")

# --- Section: Rationale Explanation ---
st.subheader("3. Rationale Explanation")
with st.expander("Step 1: Generate Explanation", expanded=st.session_state.rat_step >= 1):
    if st.button("💬 Run Step 1") or st.session_state.rat_step >= 1:
        st.session_state.rat_step = 1

        # GPT-based rationale generation using Azure OpenAI
        openai.api_type = "azure"
        openai.api_base = "https://llmwg.services.ai.azure.com/api/projects/RationaleExplainer"
        openai.api_version = "2024-03-01-preview"
        openai.api_key = "8F5cUeF5F6bt9AMBtGNi5Dy8F2FDNNGcwCYHivbw5khMZ6yYuDupJQQJ99BEACYeBjFXJ3w3AAAAACOGIcf6"

        ir_summary = "\n".join(st.session_state.get("summary_ir", []))
        vol_summary = "\n".join(st.session_state.get("summary_vol", []))

        prompt = f"""
        Based on the trade details below and predicted IFRS13 level, provide a rationale:

        Trade: {json.dumps(trade, indent=2)}
        Predicted Level: {st.session_state.get('ifrs13_level', 'N/A')}

        IR Delta Observability Summary:
        {ir_summary}

        Volatility Observability Summary:
        {vol_summary}
        """

        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing IFRS13 rationale."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            rationale = response["choices"][0]["message"]["content"]
            st.success("Rationale generated successfully")
            st.markdown(f"**Explanation:**\n\n{rationale}")
        except Exception as e:
            st.error(f"Error generating rationale: {e}")

st.markdown(get_workflow_html_rat(st.session_state.rat_step), unsafe_allow_html=True)
