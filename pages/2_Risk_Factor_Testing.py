import streamlit as st
import pandas as pd
import os
from Observability_Stress_Module import simulate_greeks, generate_trade_pv_and_risk_pvs, ir_delta_stress_test, vol_risk_stress_test
from workflow_styles import get_workflow_css, get_workflow_html_rf

st.set_page_config(page_title="Risk Factor Testing", layout="wide")
st.markdown(get_workflow_css(), unsafe_allow_html=True)
st.title("2️⃣ Risk Factor Observability Testing")

# --- Load secrets from environment variables as fallback ---
def get_secret(key, default=""):
    return os.getenv(key, st.secrets.get(key, default))

# --- Trade Input ---
st.sidebar.header("Trade Details")
product_type = st.sidebar.selectbox("Product Type", ["IR Swaption", "Bond", "CapFloor", "IRSwap"], index=0)
notional = st.sidebar.number_input("Notional", min_value=1_000_000, step=1_000_000, value=10_000_000)
currency = st.sidebar.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
option_type = st.sidebar.selectbox("Option Type", ["Receiver", "Payer"])
strike = st.sidebar.slider("Strike (%)", 0.0, 10.0, 2.5, 0.1)
expiry_tenor = st.sidebar.selectbox("Expiry Tenor (Y)", [2, 3, 5, 10])
maturity_tenor = st.sidebar.selectbox("Maturity Tenor (Y)", [5, 10, 15, 20, 30])

trade = {
    "product_type": product_type,
    "currency": currency,
    "option_type": option_type,
    "expiry_tenor": expiry_tenor,
    "maturity_tenor": maturity_tenor,
    "strike": strike,
    "notional": notional
}
st.session_state["trade"] = trade

st.markdown(get_workflow_html_rf(4 if st.session_state.get("rf_done") else 0), unsafe_allow_html=True)

if st.button("▶ Run Risk Factor Inference"):
    greeks = simulate_greeks(trade)
    trade["trade_pv"], generated_pvs = generate_trade_pv_and_risk_pvs(greeks)
    greeks.update(generated_pvs)

    ir_stressed, ir_report, ir_stress_pv, ir_msgs = ir_delta_stress_test(trade, greeks)
    vol_stressed, vol_report, vol_stress_pv, vol_msgs = vol_risk_stress_test(trade, greeks)
    total_stress_pv = ir_stress_pv + vol_stress_pv
    final_level = "Level 3" if total_stress_pv > 0.1 * trade["trade_pv"] else "Level 2"

    st.session_state.update({
        "greeks": greeks,
        "generated_pvs": generated_pvs,
        "ir_summary": ir_msgs,
        "vol_summary": vol_msgs,
        "ir_report_df": pd.DataFrame(ir_report).T,
        "vol_report_df": pd.DataFrame(vol_report).T,
        "trade_pv": trade["trade_pv"],
        "ir_stress_pv": ir_stress_pv,
        "vol_stress_pv": vol_stress_pv,
        "final_level": final_level,
        "rf_done": True
    })
    st.rerun()

# Show results
if st.session_state.get("rf_done"):
    st.subheader("Greeks and PV Contributions")
    st.dataframe(pd.DataFrame.from_dict(st.session_state["greeks"], orient="index", columns=["Value"]))
    st.dataframe(pd.DataFrame.from_dict(st.session_state["generated_pvs"], orient="index", columns=["PV"]))

    st.subheader("IR Delta Observability")
    st.dataframe(st.session_state["ir_report_df"])

    st.subheader("Volatility Observability")
    st.dataframe(st.session_state["vol_report_df"])

    st.metric("Total PV", st.session_state["trade_pv"])
    st.metric("Stress PV", st.session_state["ir_stress_pv"] + st.session_state["vol_stress_pv"])
    st.success(f"🔍 Final Observability Level: {st.session_state['final_level']}")
