import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["hour", "dayofweek", "month", "is_weekend", "temperature_c",
            "lag_1", "lag_24", "rolling_mean_24"]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.sort_values("timestamp").copy()
    ts = pd.to_datetime(d["timestamp"])
    d["hour"] = ts.dt.hour
    d["dayofweek"] = ts.dt.dayofweek
    d["month"] = ts.dt.month
    d["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    d["lag_1"] = d["energy_kwh"].shift(1)
    d["lag_24"] = d["energy_kwh"].shift(24)
    d["rolling_mean_24"] = d["energy_kwh"].shift(1).rolling(24, min_periods=1).mean()
    d["temperature_c"] = d["temperature_c"].astype(float)
    return d


def ensure_demo_data():
    path = DATA_DIR / "building_energy.csv"
    if path.exists():
        return
    rng = np.random.default_rng(17)
    times = pd.date_range("2025-01-01", periods=24 * 180, freq="h")
    rows = []
    for bid, base, area in [("Campus-A", 24, 1.0), ("Office-B", 15, 0.72), ("Lab-C", 32, 1.15)]:
        hour = times.hour.to_numpy()
        weekday = times.dayofweek.to_numpy() < 5
        daily = 7 * np.sin((hour - 7) * np.pi / 12) + 8
        occupancy = np.where(weekday & (hour >= 8) & (hour < 19), 1.0, 0.28)
        temperature = 27 + 6 * np.sin(2 * np.pi * (times.dayofyear.to_numpy() - 80) / 365) + 3 * np.sin(2*np.pi*(hour-14)/24)
        cooling = np.maximum(temperature - 23, 0) * 0.45
        noise = rng.normal(0, 1.8, len(times))
        energy = np.maximum(2, base * occupancy + daily * area + cooling * area + noise)
        # Inject a few synthetic abnormal periods to exercise anomaly detection.
        for start in [24 * 33 + 2, 24 * 101 + 1, 24 * 145 + 3]:
            energy[start:start+4] += rng.uniform(8, 13)
        for i, ts in enumerate(times):
            rows.append((bid, ts, float(energy[i]), float(temperature[i]), float(area * 2500)))
    generated = pd.DataFrame(rows, columns=["building_id", "timestamp", "energy_kwh", "temperature_c", "floor_area_m2"])
    generated["data_source"] = "synthetic demo"
    generated.to_csv(path, index=False)
    demo_path = DATA_DIR / "demo_building_energy.csv"
    if not demo_path.exists():
        generated.to_csv(demo_path, index=False)


def load_data():
    df = pd.read_csv(DATA_DIR / "building_energy.csv", parse_dates=["timestamp"])
    if "building_id" not in df.columns:
        df["building_id"] = "Imported-Building"
    if "temperature_c" not in df.columns:
        df["temperature_c"] = 24.0
    if "data_source" not in df.columns:
        df["data_source"] = "imported CSV"
    required = {"building_id", "timestamp", "energy_kwh", "temperature_c"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df.dropna(subset=["timestamp", "energy_kwh"]).sort_values(["building_id", "timestamp"])


def train_models(df: pd.DataFrame):
    parts = []
    for _, group in df.groupby("building_id"):
        parts.append(add_features(group))
    feat = pd.concat(parts).dropna(subset=["lag_1", "lag_24"]).copy()
    # Chronological split: latest 20% is held out from training.
    feat = feat.sort_values("timestamp")
    cutoff = feat["timestamp"].quantile(0.8)
    train = feat[feat.timestamp <= cutoff]
    test = feat[feat.timestamp > cutoff]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(n_estimators=180, max_depth=3, learning_rate=0.04, random_state=17),
    )
    model.fit(train[FEATURES], train["energy_kwh"])
    pred = model.predict(test[FEATURES])
    mae = float(mean_absolute_error(test["energy_kwh"], pred))
    rmse = float(np.sqrt(mean_squared_error(test["energy_kwh"], pred)))
    joblib.dump(model, MODEL_DIR / "forecast.joblib")

    anomaly_features = feat[["energy_kwh", "temperature_c", "hour", "dayofweek"]].copy()
    anomaly = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(contamination=0.025, random_state=17),
    )
    anomaly.fit(anomaly_features)
    joblib.dump(anomaly, MODEL_DIR / "anomaly.joblib")
    return {"forecast_test_MAE_kWh": round(mae, 3), "forecast_test_RMSE_kWh": round(rmse, 3),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "note": "Metrics are from the included synthetic demonstration data."}


def _future_feature_row(history: pd.DataFrame, ts: pd.Timestamp, predicted_values: list):
    energies = history["energy_kwh"].astype(float).tolist() + predicted_values
    hour = ts.hour
    return {
        "hour": hour,
        "dayofweek": ts.dayofweek,
        "month": ts.month,
        "is_weekend": int(ts.dayofweek >= 5),
        "temperature_c": float(history["temperature_c"].tail(24).mean()),
        "lag_1": energies[-1],
        "lag_24": energies[-24] if len(energies) >= 24 else float(np.mean(energies)),
        "rolling_mean_24": float(np.mean(energies[-24:])),
    }


def forecast_next_24h(building_data: pd.DataFrame):
    model_path = MODEL_DIR / "forecast.joblib"
    if not model_path.exists():
        train_models(load_data())
    model = joblib.load(model_path)
    history = building_data.sort_values("timestamp").copy()
    last_ts = pd.to_datetime(history["timestamp"].iloc[-1])
    predictions = []
    times = []
    for step in range(1, 25):
        ts = last_ts + pd.Timedelta(hours=step)
        row = _future_feature_row(history, ts, predictions)
        pred = float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])
        pred = max(0.0, pred)
        predictions.append(pred)
        times.append(ts)
    out = pd.DataFrame({"timestamp": times, "predicted_kwh": predictions})
    out.attrs["model"] = "GradientBoostingRegressor trained on historical data"
    return out


def detect_anomalies(building_data: pd.DataFrame):
    model = joblib.load(MODEL_DIR / "anomaly.joblib")
    d = building_data.copy()
    ts = pd.to_datetime(d.timestamp)
    features = pd.DataFrame({
        "energy_kwh": d.energy_kwh.astype(float),
        "temperature_c": d.temperature_c.astype(float),
        "hour": ts.dt.hour,
        "dayofweek": ts.dt.dayofweek,
    })
    d["anomaly"] = model.predict(features)
    d["anomaly_score"] = model.decision_function(features)
    return d


def tariff_vector(n: int, base_tariff: float, profile: str):
    if profile == "Peak / off-peak":
        # Illustrative tariff: higher during 17:00–21:00; lower overnight.
        return np.array([base_tariff * (1.6 if 17 <= h < 21 else 0.75 if h < 6 else 1.0)
                         for h in range(n)], dtype=float)
    return np.full(n, base_tariff, dtype=float)


def estimate_cost(kwh, base_tariff: float, profile: str):
    return float(np.sum(np.asarray(kwh, dtype=float) * tariff_vector(len(kwh), base_tariff, profile)))


def optimize_schedule(forecast: pd.DataFrame, load_kw: float, required_hours: int,
                      max_total_kw: float, tariff: float, tariff_profile: str,
                      allowed_start: int = 0, allowed_end: int = 24,
                      comfort_min: float = 22.0, comfort_max: float = 26.0):
    f = forecast.copy().reset_index(drop=True)
    prices = tariff_vector(len(f), tariff, tariff_profile)
    # Select lowest-price hours that also satisfy the configured total-power limit.
    hours = pd.to_datetime(f["timestamp"]).dt.hour.to_numpy()
    within_window = (hours >= allowed_start) & (hours < allowed_end)
    # Comfort bounds are carried into the agent configuration for reporting. This
    # simplified scheduler does not claim to model indoor temperature or HVAC physics.
    feasible = ((f["predicted_kwh"].to_numpy() + load_kw) <= max_total_kw) & within_window
    candidates = np.where(feasible)[0]
    candidates = sorted(candidates, key=lambda i: (prices[i], f["predicted_kwh"].iloc[i]))
    chosen = candidates[:required_hours]
    if len(chosen) < required_hours:
        return pd.DataFrame(columns=["timestamp", "predicted_base_kwh", "scheduled_load_kw", "tariff"])
    scheduled = np.zeros(len(f))
    scheduled[chosen] = load_kw
    return pd.DataFrame({
        "timestamp": f["timestamp"],
        "predicted_base_kwh": f["predicted_kwh"],
        "scheduled_load_kw": scheduled,
        "tariff": prices,
    })


def normalize_uploaded_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common smart-meter CSV column names into the app schema."""
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    aliases = {
        "timestamp": ["timestamp", "datetime", "date_time", "date", "time", "reading_time"],
        "energy_kwh": ["energy_kwh", "kwh", "consumption_kwh", "energy", "usage_kwh", "electricity_kwh"],
        "building_id": ["building_id", "building", "site", "meter_id", "location"],
        "temperature_c": ["temperature_c", "temperature", "temp_c", "outdoor_temperature"],
    }
    rename = {}
    for target, names in aliases.items():
        if target not in d.columns:
            found = next((n for n in names if n in d.columns), None)
            if found:
                rename[found] = target
    d = d.rename(columns=rename)
    if "timestamp" not in d.columns or "energy_kwh" not in d.columns:
        raise ValueError("CSV needs a timestamp/datetime column and an energy_kwh/kWh column.")
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["energy_kwh"] = pd.to_numeric(d["energy_kwh"], errors="coerce")
    d = d.dropna(subset=["timestamp", "energy_kwh"])
    if d.empty:
        raise ValueError("No valid timestamped numeric energy readings found.")
    if (d["energy_kwh"] < 0).any():
        raise ValueError("Energy readings cannot be negative. Check whether your file contains net/export readings.")
    if "building_id" not in d.columns:
        d["building_id"] = "Imported-Building"
    d["building_id"] = d["building_id"].fillna("Imported-Building").astype(str)
    if "temperature_c" not in d.columns:
        d["temperature_c"] = 24.0
    d["temperature_c"] = pd.to_numeric(d["temperature_c"], errors="coerce").fillna(24.0)
    d["data_source"] = "imported CSV"
    return d.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def save_uploaded_data(df: pd.DataFrame):
    """Save normalized uploaded data as the active dataset."""
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "building_energy.csv", index=False)


def build_context(data: pd.DataFrame, forecast: pd.DataFrame, anomalies: pd.DataFrame,
                  tariff: float, profile: str = "Peak / off-peak") -> str:
    d = data.sort_values("timestamp")
    flagged = anomalies[anomalies["anomaly"] == -1]
    last = d.tail(24)
    previous = d.iloc[-48:-24] if len(d) >= 48 else d.head(max(1, len(d) - len(last)))
    recent_avg = float(last.energy_kwh.mean()) if len(last) else 0.0
    prev_avg = float(previous.energy_kwh.mean()) if len(previous) else recent_avg
    delta = ((recent_avg - prev_avg) / prev_avg * 100.0) if prev_avg else 0.0
    return (
        f"Building: {d.building_id.iloc[-1]}; rows: {len(d)}; "
        f"time span: {d.timestamp.min()} to {d.timestamp.max()}; "
        f"latest reading: {float(d.energy_kwh.iloc[-1]):.2f} kWh at {d.timestamp.iloc[-1]}; "
        f"last 24 readings average: {recent_avg:.2f} kWh; preceding-period average: {prev_avg:.2f} kWh; "
        f"recent average change: {delta:+.1f}%; total anomaly flags in dataset: {len(flagged)}; "
        f"next 24h forecast total: {float(forecast.predicted_kwh.sum()):.2f} kWh; "
        f"forecast average: {float(forecast.predicted_kwh.mean()):.2f} kWh; "
        f"tariff input: {tariff} currency/kWh; tariff profile: {profile}. "
        "Energy units are assumed to be kWh per interval as provided by the CSV. Data does not include equipment state, occupancy, or verified indoor temperatures unless supplied separately."
    )


def answer_energy_question(question: str, context: str, history=None):
    """Answer common building-energy questions locally, without an API key or internet."""
    import re

    q = (question or "").lower()

    def number(pattern, default=None):
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if not match:
            return default
        try:
            return float(match.group(1).replace(",", ""))
        except (TypeError, ValueError):
            return default

    recent = number(r"last 24 readings average:\s*([+-]?[0-9,.]+)")
    previous = number(r"preceding-period average:\s*([+-]?[0-9,.]+)")
    change = number(r"recent average change:\s*([+-]?[0-9,.]+)%")
    anomalies = number(r"total anomaly flags in dataset:\s*([0-9,.]+)", 0)
    forecast_total = number(r"next 24h forecast total:\s*([0-9,.]+)")
    forecast_avg = number(r"forecast average:\s*([0-9,.]+)")

    if any(word in q for word in ("increase", "increased", "higher", "spike", "rise", "why", "more", "decrease", "decreased")):
        if change is not None and recent is not None and previous is not None:
            if change > 1:
                trend = f"The latest 24-reading average is {recent:.2f} kWh, versus {previous:.2f} kWh in the preceding period ({change:+.1f}%). That confirms an increase in this dataset."
            elif change < -1:
                trend = f"The latest 24-reading average is {recent:.2f} kWh, versus {previous:.2f} kWh in the preceding period ({change:+.1f}%). That is a decrease, not an increase, in this dataset."
            else:
                trend = f"The latest 24-reading average is {recent:.2f} kWh, versus {previous:.2f} kWh in the preceding period ({change:+.1f}%), so usage is broadly stable."
        else:
            trend = "I could not calculate the recent trend from the available summary."
        causes = (
            "Possible causes to investigate include HVAC/AC runtime and setpoints, occupancy and operating hours, "
            "equipment and lighting schedules, outdoor temperature/weather, and meter/data-quality issues."
        )
        anomaly_note = f" The anomaly detector flagged {int(anomalies)} readings in the full dataset; these are investigation candidates, not confirmed faults." if anomalies is not None else ""
        return f"{trend}\n\n{causes} The available summary does not identify which cause applies.{anomaly_note}\n\nFor a stronger diagnosis, compare hourly readings against the same hours on previous days and add HVAC runtime, occupancy, and outdoor-temperature data.\n\nData summary: {context}", False

    if any(word in q for word in ("forecast", "tomorrow", "predict", "next 24", "expected")):
        forecast_text = "The forecast values are unavailable in the supplied context."
        if forecast_total is not None and forecast_avg is not None:
            forecast_text = f"The model forecasts {forecast_total:.2f} kWh over the next 24 hours, averaging {forecast_avg:.2f} kWh per interval. This is a prediction, not measured consumption."
        return forecast_text + " Forecast accuracy depends on the history and whether relevant drivers are represented.\n\nData summary: " + context, False

    if any(word in q for word in ("anomal", "alert", "unusual", "fault", "problem")):
        return f"The detector flagged {int(anomalies or 0)} readings across the available dataset. Review their timestamps and compare them with meter records, equipment schedules, occupancy, and weather. An anomaly is not proof of equipment failure.\n\nData summary: {context}", False

    if any(word in q for word in ("cost", "tariff", "price", "save", "schedule")):
        return "Use the Cost optimizer tab to compare flexible-load schedules under the configured tariff and permitted operating hours. Confirm the real utility tariff and equipment constraints first. The schedule is a recommendation only; this prototype does not control equipment.\n\nData summary: " + context, False

    return "I can answer common usage, forecast, anomaly, and scheduling questions locally. For a cause diagnosis, the energy readings show trends but usually cannot prove the cause without occupancy, weather, and equipment-state data.\n\nData summary: " + context, False
