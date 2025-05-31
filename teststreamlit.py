import streamlit as st

st.markdown("""
<style>
.stepper-container {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 3px 10px;
  font-size: 9px;
  font-weight: 500;
  color: white;
  border-radius: 9999px;
  min-width: 110px;
  text-align: center;
  line-height: 1.4;
  box-shadow: 0 1px 2px rgba(0,0,0,0.08);
}

.pill.llm { background-color: #00A3C4; }
.pill.func { background-color: #b19cd9; }
.pill.success { background-color: #28a745; }

.arrow {
  font-size: 14px;
  color: #ccc;
  user-select: none;
}
</style>

<div class="stepper-container">
  <span class="pill func"><span class="pill-icon">🛠</span>Function</span>
  <span class="arrow">➔</span>
  <span class="pill llm"><span class="pill-icon">🧠</span>LLM: GPT-4</span>
  <span class="arrow">➔</span>
  <span class="pill success"><span class="pill-icon">✔️</span>Completed</span>
</div>
""", unsafe_allow_html=True)
