# EnergiAI — Building Energy Optimization Agent

A local Streamlit prototype combining machine-learning forecasts, anomaly alerts, tariff-aware flexible-load scheduling, CSV ingestion, and an offline local conversational assistant.

## Run on Windows

1. Install Python 3.10 or newer.
2. Open Command Prompt and go into the extracted project folder (the folder containing `app.py`).
3. Install dependencies:

   ```bat
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bat
   python -m streamlit run app.py
   ```

5. Open the `Local URL` shown in the terminal, usually `http://localhost:8501`.

## Bring your own historical energy data

In the sidebar, upload a CSV and click **Use uploaded CSV**. Required columns:

- `timestamp` (or `datetime`, `date_time`, `date`, `time`, `reading_time`)
- `energy_kwh` (or `kwh`, `consumption_kwh`, `energy`, `usage_kwh`, `electricity_kwh`)

Optional columns:

- `building_id` (or `building`, `site`, `meter_id`, `location`)
- `temperature_c` (or `temperature`, `temp_c`, `outdoor_temperature`)

Example:

```csv
timestamp,building_id,energy_kwh,temperature_c
2026-10-01 09:00:00,Office-A,18.7,29.1
2026-10-01 10:00:00,Office-A,20.3,30.0
```

The app normalizes these columns and saves the active dataset in `data/building_energy.csv`. Keep timestamps consistent and ensure each reading's energy unit/interval is understood. For good forecasts, provide several weeks or months of regular readings if possible. The app retrains models after data changes that remove existing model files; you can also press **Train / retrain models**.

## Enable the LLM chat

The chat tab works in local fallback mode without an API key. To enable OpenAI responses, set an API key in your Windows Command Prompt before starting Streamlit:

```bat
set OPENAI_API_KEY=your_api_key_here
set OPENAI_MODEL=gpt-4o-mini
python -m streamlit run app.py
```

Keep the key private; do not paste it into source code or commit it to a repository. API usage may incur charges. If no key is configured, the app uses a deterministic local assistant that summarizes available data and limitations rather than calling an LLM.

## Refreshing data / live meter integration

The **Auto-refresh dashboard** option reloads the active CSV from disk every 15 seconds to 5 minutes. It does not itself connect to a smart meter. For near-real-time data, configure your meter/BMS integration or a separate secure ingestion process to append new readings to `data/building_energy.csv` using the required schema. Use an authenticated site-specific connector (for example, the vendor's documented API or a building-management gateway); credentials should be stored in environment variables or a secret manager.

## Current capabilities and limitations

- Gradient Boosting model forecasts the next 24 hours from historical consumption and timestamp features.
- Isolation Forest flags unusual readings for review.
- A simple optimizer chooses low-cost eligible hours for a flexible load, subject to a configured building load limit and allowed time window.
- The local assistant uses the selected building data and clearly distinguishes measured trends from possible causes.
- Comfort temperature bounds are captured as operating settings, but this version does **not** simulate HVAC thermodynamics or verify indoor comfort. It does not control equipment automatically.
- Peak/off-peak tariff periods are illustrative; replace them with the actual utility tariff before using cost results.
- Uploading CSV is supported. Direct smart-meter/BMS connectors need to be implemented for the actual device/vendor and site security requirements.
- Test and validate model performance on representative real data before relying on decisions.
