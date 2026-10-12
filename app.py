import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from energi import (
    DATA_DIR, MODEL_DIR, ensure_demo_data, load_data, train_models,
    forecast_next_24h, detect_anomalies, optimize_schedule, estimate_cost,
    normalize_uploaded_csv, save_uploaded_data, build_context, answer_energy_question,
)

st.set_page_config(page_title="EnergiAI", page_icon="⚡", layout="wide")
st.title("⚡ EnergiAI — Building Energy Optimization Agent")
st.caption("Forecasting • anomaly alerts • tariff-aware scheduling • conversational energy assistant")

ensure_demo_data()
with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload historical smart-meter CSV", type=["csv"], help="Required: timestamp and energy_kwh. Optional: building_id and temperature_c.")
    if uploaded is not None and st.button("Use uploaded CSV", type="primary"):
        try:
            incoming = normalize_uploaded_csv(pd.read_csv(uploaded))
            save_uploaded_data(incoming)
            for p in [MODEL_DIR / "forecast.joblib", MODEL_DIR / "anomaly.joblib"]:
                if p.exists(): p.unlink()
            st.success(f"Loaded {len(incoming):,} rows from your CSV.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load CSV: {exc}")
    if st.button("Switch back to demo data"):
        demo = DATA_DIR / "demo_building_energy.csv"
        active = DATA_DIR / "building_energy.csv"
        if demo.exists():
            import shutil
            shutil.copy2(demo, active)
        for p in [MODEL_DIR / "forecast.joblib", MODEL_DIR / "anomaly.joblib"]:
            if p.exists(): p.unlink()
        st.rerun()
    st.divider()
    st.header("Operating settings")
    tariff = st.number_input("Base electricity tariff (currency/kWh)", min_value=0.0, value=8.0, step=0.5)
    max_kw = st.number_input("Maximum building load (kW)", min_value=1.0, value=100.0, step=5.0)
    comfort_min = st.number_input("Comfort temperature minimum (°C)", value=22.0, step=0.5)
    comfort_max = st.number_input("Comfort temperature maximum (°C)", value=26.0, step=0.5)
    auto_refresh = st.checkbox("Auto-refresh dashboard", value=False, help="Reload the active CSV from disk periodically. A meter/BMS connector must write fresh readings to that file.")
    refresh_seconds = st.selectbox("Refresh interval", [15, 30, 60, 300], index=1, format_func=lambda x: f"Every {x} seconds")
    run_train = st.button("Train / retrain models")

if auto_refresh:
    st_autorefresh(interval=int(refresh_seconds * 1000), key="energi_auto_refresh")

# Keep a pristine demo file so users can switch back after importing a real dataset.
demo_path = DATA_DIR / "demo_building_energy.csv"
active_path = DATA_DIR / "building_energy.csv"
if not demo_path.exists() and active_path.exists():
    pd.read_csv(active_path).to_csv(demo_path, index=False)
df = load_data()
if run_train or not ((MODEL_DIR / "forecast.joblib").exists() and (MODEL_DIR / "anomaly.joblib").exists()):
    with st.spinner("Training models on the selected dataset..."):
        metrics = train_models(df)
    st.sidebar.success("Models trained")
    st.sidebar.json(metrics)

building = st.selectbox("Select building", sorted(df["building_id"].astype(str).unique()))
data = df[df.building_id.astype(str) == str(building)].sort_values("timestamp").copy()
if len(data) < 30:
    st.warning("For useful hourly forecasts, upload at least 30 timestamped readings; a week or more is preferable.")

is_demo = bool(data.get("data_source", pd.Series(["real"])).astype(str).eq("synthetic demo").all())
st.caption(f"Data source: {'Synthetic demonstration data' if is_demo else 'Imported CSV'} · Latest reading: {pd.to_datetime(data.timestamp).max()}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Readings", f"{len(data):,}")
c2.metric("Latest energy reading", f"{data.energy_kwh.iloc[-1]:.2f} kWh")
c3.metric("Average reading", f"{data.energy_kwh.mean():.2f} kWh")
c4.metric("Comfort target", f"{comfort_min:g}–{comfort_max:g} °C")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Forecast", "Alerts", "Cost optimizer", "Ask the AI agent"])
with tab1:
    st.plotly_chart(px.line(data.tail(24 * 14), x="timestamp", y="energy_kwh", title="Recent hourly electricity consumption"), use_container_width=True)
    st.dataframe(data.tail(20), use_container_width=True)
    if is_demo:
        st.info("You are viewing synthetic demo data. Upload your own smart-meter CSV to analyze real readings.")
    else:
        st.success("The dashboard is using readings imported from your CSV. New readings are picked up when you reload the data or re-upload an updated file.")
with tab2:
    forecast = forecast_next_24h(data)
    st.plotly_chart(px.line(forecast, x="timestamp", y="predicted_kwh", title="Predicted energy use for the next 24 hours"), use_container_width=True)
    st.metric("Next 24h predicted energy", f"{forecast.predicted_kwh.sum():.1f} kWh")
    st.caption("The model is trained on the selected dataset. Forecast quality depends on data volume, consistency, and whether relevant drivers are available.")
with tab3:
    anomalies = detect_anomalies(data)
    flagged = anomalies[anomalies.anomaly == -1].copy()
    st.metric("Unusual readings flagged", len(flagged))
    if len(flagged):
        st.warning("Review these readings. An anomaly is a prompt to investigate, not proof of equipment failure.")
        st.dataframe(flagged[["timestamp", "energy_kwh", "temperature_c", "anomaly_score"]].tail(100), use_container_width=True)
    else:
        st.success("No unusual readings were flagged by the current model.")
    st.plotly_chart(px.scatter(anomalies.tail(24 * 14), x="timestamp", y="energy_kwh", color=anomalies.tail(24 * 14).anomaly.astype(str), title="Anomaly detection (−1 = unusual, 1 = typical)"), use_container_width=True)
with tab4:
    st.markdown("Schedule a flexible load, such as a non-urgent process, only during hours it is allowed to run.")
    col1, col2 = st.columns(2)
    with col1:
        load_kw = st.number_input("Flexible equipment power (kW)", min_value=0.1, max_value=100.0, value=3.0, step=0.5)
        duration = st.slider("Required run time (hours in next 24h)", 1, 12, 4)
        profile = st.selectbox("Tariff type", ["Flat", "Peak / off-peak"])
    with col2:
        allowed_start = st.slider("Earliest allowed start hour", 0, 23, 0)
        allowed_end = st.slider("Latest allowed end hour (exclusive)", 1, 24, 24)
        st.caption("Comfort bounds are shown above as operating constraints; this prototype does not control HVAC or simulate room temperatures.")
    schedule = optimize_schedule(forecast, float(load_kw), int(duration), float(max_kw), float(tariff), profile,
                                 allowed_start=int(allowed_start), allowed_end=int(allowed_end),
                                 comfort_min=float(comfort_min), comfort_max=float(comfort_max))
    if schedule.empty:
        st.error("No feasible schedule found within the selected hours and building power limit. Try a smaller load, longer availability window, or higher limit.")
    else:
        chosen = schedule[schedule.scheduled_load_kw > 0]
        base_cost = estimate_cost(forecast.predicted_kwh.to_numpy(), tariff, profile)
        combined = forecast.predicted_kwh.to_numpy() + schedule.scheduled_load_kw.to_numpy()
        combined_cost = estimate_cost(combined, tariff, profile)
        a, b, c = st.columns(3)
        a.metric("Forecast base cost", f"{base_cost:.2f}")
        b.metric("Cost with scheduled load", f"{combined_cost:.2f}")
        c.metric("Flexible load energy", f"{chosen.scheduled_load_kw.sum():.1f} kWh")
        st.plotly_chart(px.bar(schedule, x="timestamp", y="scheduled_load_kw", title="Recommended flexible-load schedule"), use_container_width=True)
        st.dataframe(chosen, use_container_width=True)
        st.write("Recommended hours: " + ", ".join(pd.to_datetime(chosen.timestamp).dt.strftime("%a %H:%M")))
        st.caption("Cost estimates are only as accurate as the tariff entered. Peak/off-peak hours are illustrative and must be replaced with your utility's tariff. No equipment is controlled automatically.")
with tab5:
    st.markdown("Ask about the selected building's recent usage, forecast, anomalies, and scheduling options.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    question = st.chat_input("Example: Why did energy usage increase recently?")
    if question:
        context = build_context(data, forecast_next_24h(data), detect_anomalies(data), tariff, profile="Peak / off-peak")
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"): st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing building data..."):
                response, used_llm = answer_energy_question(question, context, st.session_state.chat_history[:-1])
            st.markdown(response)
            st.caption("Local data-grounded assistant — no API key required")
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

st.divider()
st.caption("Prototype safety: recommendations only. It does not connect to meters/BMS APIs or actuate equipment. CSV refresh is manual; production deployment needs a secure meter connector, scheduled ingestion, validated forecasts, and site-specific comfort/control constraints.")
