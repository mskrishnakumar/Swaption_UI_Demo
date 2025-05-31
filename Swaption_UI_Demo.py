import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import date
from openai import AzureOpenAI
from Observability_Stress_Module import (
    run_full_observability_stress_test,
    simulate_greeks,
    ir_delta_stress_test,
    vol_risk_stress_test,
    generate_trade_pv_and_risk_pvs,
    ois_curve_map
)
from workflow_styles import (
    get_workflow_css,
    get_workflow_html_ml,
    get_workflow_html_rf,
    get_workflow_html_rat
)

# --- Initialize Session State ---
for key in ["ml_done", "rf_done", "rat_done"]:
    if key not in st.session_state:
        st.session_state[key] = False

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

# --- Azure GPT-4o Client ---
def get_rationale_from_gpt(ir_summary, vol_summary, model_pred):
    client = AzureOpenAI(
        api_key=st.secrets["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst who explains IFRS 13 classification decisions "
                "clearly and concisely based on input data. Do not include background information "
                "about IFRS 13. Focus strictly on justifying the classification result."
            )
        },
        {
            "role": "user",
            "content": (
                f"IR Delta Observability Summary: {ir_summary}\n\n"
                f"Volatility Observability Summary: {vol_summary}\n\n"
                f"Model Predicted Level: {model_pred}\n\n"
                "Based on the data above, provide a concise justification for the predicted IFRS 13 level. "
                "Keep the response brief, technical, and to the point."
            )
        }
    ]
    response = client.chat.completions.create(
        model=st.secrets["AZURE_OPENAI_MODEL"],
        messages=messages,
        temperature=0.5
    )
    return response.choices[0].message.content


# --- Section: Machine Learning Model Prediction ---
with st.container(border=True):
    st.subheader("1. Machine Learning Model Prediction")
    step = 3 if st.session_state.ml_done else 0
    st.markdown(get_workflow_html_ml(step), unsafe_allow_html=True)

    if st.button("\u25B6 Run ML Inference Workflow"):
        st.code(json.dumps(trade, indent=2), language='json')
        st.success("\u2705 ML Input ready")
        st.info("Features extracted successfully (mock)")

        model_pred = mock_model_prediction(trade)
        st.session_state["ifrs13_level"] = model_pred
        st.session_state["model_pred"] = model_pred
        st.success("\u2705 ML pipeline executed")
        st.success(f"Predicted IFRS13 Level: {model_pred}")

        level_html = f"""
            <div style='background-color:#d4edda;color:#155724;padding:10px;
            border-left:5px solid #28a745;border-radius:5px;margin-top:20px;
            font-weight:bold'>IFRS13 Level:<br>{model_pred}</div>
        """
        st.sidebar.markdown(level_html, unsafe_allow_html=True)

        st.session_state.ml_done = True
        #st.rerun()

# --- Section: Risk Factor-based Inference ---
with st.container(border=True):
    st.subheader("2. Risk Factor Observability testing")
    step = 4 if st.session_state.rf_done else 0
    st.markdown(get_workflow_html_rf(step), unsafe_allow_html=True)

    if st.button("\u25B6 Run Risk Factor Inference Workflow"):
        greeks = simulate_greeks(trade)
        st.success("\u2705 Risk factors simulated")

        trade["trade_pv"], generated_pvs = generate_trade_pv_and_risk_pvs(greeks)
        greeks.update(generated_pvs)
        st.success("\u2705 IR Delta Observability Test Completed")

        ir_stressed, ir_report, ir_stress_pv, ir_msgs = ir_delta_stress_test(trade, greeks)
        st.session_state["ir_summary"] = ir_msgs
        st.dataframe(pd.DataFrame(ir_report).T)

        st.success("\u2705 Volatility Observability Test Completed")
        vol_stressed, vol_report, vol_stress_pv, vol_msgs = vol_risk_stress_test(trade, greeks)
        st.session_state["vol_summary"] = vol_msgs
        st.dataframe(pd.DataFrame(vol_report).T)

        total_stress_pv = ir_stress_pv + vol_stress_pv
        final_level = "Level 3" if total_stress_pv > 0.1 * trade["trade_pv"] else "Level 2"
        st.metric("Total Stress PV", total_stress_pv)

        if final_level == "Level 3":
            st.error("\U0001F534 Unobservable stress exceeds threshold → Level 3")
        else:
            st.success("\U0001F7E2 Within threshold → Level 2")

        st.session_state["final_level"] = final_level
        st.session_state.rf_done = True
        #st.rerun()

# --- Section: Rationale Explanation ---
with st.container(border=True):
    st.subheader("3. Rationale Explanation")
    step = 1 if st.session_state.rat_done else 0
    st.markdown(get_workflow_html_rat(step), unsafe_allow_html=True)

    if st.button("\u25B6 Run Rationale Generation Workflow"):
        if all(k in st.session_state for k in ["ir_summary", "vol_summary", "model_pred"]):
            rationale = get_rationale_from_gpt(
                ir_summary="\n".join(st.session_state["ir_summary"]),
                vol_summary="\n".join(st.session_state["vol_summary"]),
                model_pred=st.session_state["model_pred"]
            )
            st.success("\u2705 Rationale explanation generated below")
            st.markdown(
                f"""<div style='background-color:#f1f1f1;color:#000000;padding:10px;
                border-left:5px solid #0078D7;border-radius:5px'>
                <b>Explanation:</b><br>{rationale}</div>""",
                unsafe_allow_html=True
            )

            st.session_state.rat_done = True
            #st.rerun()
        else:
            st.warning("Run both ML and Risk Factor workflows before generating rationale.")
