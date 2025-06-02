import streamlit as st
import pandas as pd
import os
import requests
import time
from streamlit_echarts import st_echarts
from workflow_styles import get_workflow_css, get_workflow_html_ml

st.set_page_config(page_title="ML Model Inference", layout="wide")
st.title("IFRS13 Fair Value Classification Model")
st.markdown(get_workflow_css(), unsafe_allow_html=True)
st.markdown(get_workflow_html_ml(1 if st.session_state.get("ml_done") else 0), unsafe_allow_html=True)

# --- Secret loader ---
def get_secret(key, default=""):
    return os.getenv(key, st.secrets.get(key, default))

# --- Inference Toggle ---
inference_mode = st.radio("Select Inference Mode", ["Single Trade", "Batch Inference"], horizontal=True)

# --- Single Trade Inference ---
if inference_mode == "Single Trade":
    st.header("🔍 Trade Inference")

    with st.sidebar:
        st.markdown("### 🧾 Single Trade Details")
        product_type = st.selectbox("Product Type", ["IR Swaption", "Bond", "CapFloor", "IRSwap"], index=0)
        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
        option_type = st.selectbox("Option Type", ["Receiver", "Payer"])
        notional = st.number_input("Notional", min_value=1_000_000, step=1_000_000, value=10_000_000)
        strike = st.slider("Strike (%)", 0.0, 10.0, 2.5, 0.1)
        expiry_tenor = st.selectbox("Expiry Tenor (Y)", [2, 3, 5, 10])
        maturity_tenor = st.selectbox("Maturity Tenor (Y)", [5, 10, 15, 20, 30])

    input_data = pd.DataFrame([{
        "product_type": product_type,
        "currency": currency,
        "option_type": option_type,
        "notional": notional,
        "strike": strike,
        "expiry_tenor": expiry_tenor,
        "maturity_tenor": maturity_tenor
    }])

    if st.button("▶ Run Single Trade Inference"):
        payload = {
            "input_data": {
                "columns": input_data.columns.tolist(),
                "index": [0],
                "data": input_data.values.tolist()
            }
        }
        with st.spinner("Calling ML Model..."):
            try:
                start_time = time.time()
                response = requests.post(
                    url=get_secret("AZURE_ML_ENDPOINT"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {get_secret('AZURE_ML_API_KEY')}"
                    },
                    json=payload
                )
                end_time = time.time()
                result = response.json()
                st.session_state["model_pred"] = result[0]
                st.session_state["ML_Model_elapsed_time"] = round(end_time - start_time, 2)

                st.success(f"✅ Predicted IFRS13 Level: {result[0]}")

                st.markdown("---")
                st.expander("📘 Machine Learning Model Details", expanded=True)
                st.markdown(f"⏱️ Model run completed in {st.session_state['ML_Model_elapsed_time']} seconds")
                st.subheader("📦 Input JSON Payload")
                st.json(payload, expanded=False)
                st.subheader("📘 Model details")
                st.markdown("""
                <div style='text-align: left; overflow-y: auto; max-height: 250px;
                            padding: 10px; background-color: #111111;
                            border-radius: 8px; color: #f0f0f0;
                            font-family: monospace; font-size: 14px;'>
                    <strong>Model:</strong> Gradient Boosting (AutoML)<br>
                    <strong>Version:</strong> Gradient Boosting (AutoML)<br>
                    <strong>Trained on:</strong> Synthetic IR Swaption Trades<br>
                    <strong>Features:</strong> product_type, currency, option_type, notional, strike, expiry_tenor, maturity_tenor<br>
                    <strong>Accuracy:</strong> 86.2%<br>
                    <strong>AUC:</strong> 0.74<br>
                    <strong>Last Trained:</strong> 01-Jun-2025<br><br>
                    This model predicts IFRS13 Level based on risk and trade characteristics.<br>
                    AutoML handled class imbalance and feature engineering.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Model call failed: {e}")

# --- Batch Inference ---
with st.expander("📥 Batch Inference - Multiple Trades", expanded=(inference_mode == "Batch Inference")):
    if inference_mode == "Batch Inference":
        st.header("📥 Batch Inference - Multiple Trades")

        st.markdown("""
        Upload a CSV file containing multiple trades to run bulk IFRS13 model inference. 
        Each row should include the following columns:
        - product_type
        - currency
        - option_type
        - notional
        - strike
        - expiry_tenor
        - maturity_tenor

        📌 Optional: Include additional columns like `trading_desk` for visualizations.
        """)

        with st.sidebar:
            st.markdown("### 📋 Batch Trade Summary")
            uploaded_file = st.file_uploader("Upload CSV for Model Inference", type=["csv"], key="batch")

        if uploaded_file:
            df_infer = pd.read_csv(uploaded_file)
            required_cols = ["product_type", "currency", "option_type", "notional", "strike", "expiry_tenor", "maturity_tenor"]

            if all(col in df_infer.columns for col in required_cols):
                payload = {
                    "input_data": {
                        "columns": required_cols,
                        "index": list(range(len(df_infer))),
                        "data": df_infer[required_cols].values.tolist()
                    }
                }
                with st.spinner("Running batch inference..."):
                    try:
                        response = requests.post(
                            url=get_secret("AZURE_ML_ENDPOINT"),
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {get_secret('AZURE_ML_API_KEY')}"
                            },
                            json=payload
                        )
                        result = response.json()
                        df_infer["Predicted IFRS13 Level"] = result
                        st.success("✅ Inference completed!")
                        st.dataframe(df_infer.head(11))
                        st.download_button("📥 Download Predicted Results", data=df_infer.to_csv(index=False), file_name="predicted_results.csv", mime="text/csv")

                    except Exception as e:
                        st.error(f"❌ Model call failed: {e}")

                # --- Development-only Visualization ---
                if "trading_desk" in df_infer.columns:
                    heatmap_data = df_infer.groupby(["trading_desk", "Predicted IFRS13 Level"]).size().reset_index(name="count")
                    rows = heatmap_data["trading_desk"].unique().tolist()
                    cols = heatmap_data["Predicted IFRS13 Level"].unique().tolist()

                    row_map = {v: i for i, v in enumerate(rows)}
                    col_map = {v: i for i, v in enumerate(cols)}

                    data = [[col_map[c], row_map[r], int(v)] for r, c, v in heatmap_data.values]

                    option = {
                        "tooltip": {"position": "top"},
                        "grid": {"height": "50%", "top": "10%"},
                        "xAxis": {"type": "category", "data": cols, "splitArea": {"show": True}},
                        "yAxis": {"type": "category", "data": rows, "splitArea": {"show": True}},
                        "visualMap": {
                            "min": 0,
                            "max": max(heatmap_data["count"]),
                            "calculable": True,
                            "orient": "horizontal",
                            "left": "center",
                            "bottom": "15%",
                        },
                        "series": [
                            {
                                "name": "Trade Count",
                                "type": "heatmap",
                                "data": data,
                                "label": {"show": True},
                                "emphasis": {
                                    "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
                                },
                            }
                        ],
                    }

                    st.subheader("📊 ECharts Heatmap: IFRS13 Level vs Trading Desk")
                    st_echarts(option, height="400px")

            else:
                st.warning(f"CSV must contain: {', '.join(required_cols)}")
