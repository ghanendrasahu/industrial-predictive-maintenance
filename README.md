# 🏭 Industrial Predictive Maintenance

Turn raw machine telemetry into a maintenance decision: **how likely is this
unit to fail, when, and what should the crew do about it?**

```
Sensors            temperature · vibration · pressure · rpm · voltage · current
   ↓
Time-series        rolling window: mean · std · slope · last
   ↓
Anomaly            IsolationForest  → is the current state unusual?
   ↓
Failure model      calibrated GBM   → P(failure within the planning horizon)
   ↓
Remaining life     RandomForest     → hours/days until the next failure
   ↓
Decision           risk band + primary indicators + recommended action
```

## The decision it produces

```json
{
  "machine_id": "M-03",
  "failure_probability": 0.835,
  "risk_band": "CRITICAL",
  "estimated_rul_hours": 120.4,
  "estimated_rul_days": 5.02,
  "primary_indicators": [
    {"sensor": "vibration_mm_s", "sigma": 3.41, "direction": "up", "anomalous": true},
    {"sensor": "temperature_c", "sigma": 2.88, "direction": "up", "anomalous": true}
  ],
  "recommended_action": "Schedule emergency inspection within 24 hours. Primary indicators: vibration_mm_s (up 3.4 sigma), temperature_c (up 2.9 sigma)."
}
```

Fleet view (`python scripts/demo.py`):

```
machine     P(failure)      band   RUL (days)  next action
--------------------------------------------------------------------------------
M-06            91.8%  CRITICAL          1.9  Schedule emergency inspection within 24 hours.
M-03            83.5%  CRITICAL          5.0  Schedule emergency inspection within 24 hours.
M-05            79.7%  CRITICAL          4.8  Schedule emergency inspection within 24 hours.
M-02            73.2%     WATCH          5.4  Inspect within 48 hours and raise monitoring frequency.
M-04             9.8%  HEALTHY         11.5  Continue routine monitoring - no action required.
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements-dev.txt

python src/simulator.py --machines 8 --cycles 900   # write data/telemetry.csv
python src/train.py --data data/telemetry.csv        # train + write models/
python scripts/demo.py                              # fleet risk report
python scripts/demo.py --machine M-03               # full payload for one unit

uvicorn app.main:app --reload --port 8000           # REST API
streamlit run app/dashboard.py                      # dashboard
```

On the bundled demo fleet the models reach **failure AUC ≈ 0.87** and
**RUL MAE ≈ 19 cycles (~19 h)** on *held-out machines* - the split is by unit,
never by row, so the numbers reflect generalising to unseen equipment.

## REST API

| Method | Route                    | Purpose                                    |
| ------ | ------------------------ | ------------------------------------------ |
| GET    | `/health`                | readiness (model loaded, rows, machines)   |
| GET    | `/api/machines`          | fleet risk ranking, most urgent first      |
| GET    | `/api/predict/{id}`      | full assessment for one machine            |
| POST   | `/api/refresh`           | re-read telemetry + models after a training run |

Set `API_KEY` to require an `X-API-Key` header on every `/api/*` route (401
otherwise). `BACKEND_ORIGINS` is an explicit CORS allowlist - never `*`.

```bash
curl localhost:8000/api/predict/M-03 -H "X-API-Key: $API_KEY"
```

## Telemetry contract

One row per operating hour per machine:

| column                                | unit        | meaning                          |
| ------------------------------------- | ----------- | -------------------------------- |
| `machine_id`                          | -           | unit identifier                  |
| `cycle`                               | hours       | operating hour index             |
| `temperature_c`, `vibration_mm_s`,    | °C, mm/s    | thermal / vibration signature    |
| `pressure_bar`, `rpm`, `voltage_v`,   | bar, rpm, V | process + electrical state       |
| `current_a`                           | A           | motor load                       |
| `rul`, `failed`, `next_fault_cycle`   | -           | labels (training only)           |

`src/simulator.py` generates a reproducible fleet (seeded) with realistic
degradation physics: vibration and temperature climb super-linearly as a
failure approaches, current rises with friction, pressure sags, rpm droops and
flutters. A machine that fails is **repaired and returns to service**, so the
fleet contains units at every stage of wear - and the failure a model must
predict always lies *ahead* of the observed telemetry.

## Models

| model           | algorithm                     | question it answers                        |
| --------------- | ----------------------------- | ------------------------------------------ |
| `anomaly`       | IsolationForest               | is this state unusual for the fleet?       |
| `failure`       | calibrated GradientBoosting   | will it fail within the horizon? (7 days)  |
| `rul`           | RandomForestRegressor         | how many hours of life are left?           |

Probabilities are calibrated (sigmoid), so `83.5%` means something.

## Development

```bash
ruff check .            # lint
python -m pytest tests  # 27 offline, seeded tests
docker build -t pdm .   # two-stage, non-root, HEALTHCHECK on /health
```

CI runs lint, the test suite, an end-to-end simulate→train→score smoke, and a
Docker build that asserts the runtime image has no compiler and answers
`/health` as a non-root user.

## Project layout

```
src/simulator.py    seeded telemetry with degradation physics
src/features.py     rolling-window feature builder
src/train.py        anomaly + failure + RUL training (machine-level holdout)
src/predict.py      assessment: probability, RUL, indicators, action
app/main.py         FastAPI service (API-key gated, CORS allowlist)
app/dashboard.py    Streamlit fleet dashboard
scripts/demo.py     CLI risk report
tests/              offline, deterministic test suite
```

## Production notes

* Replace `src/simulator.py` with a real historian (OPC-UA / MQTT / SQL).
* Labels come from maintenance records: `failed` is the repair timestamp,
  `rul` = next repair - current time.
* Re-train on a schedule, call `POST /api/refresh`, and alert on `risk_band`
  transitions rather than on absolute probabilities.
* Retrain per equipment class - bearings, pumps and compressors do not share
  a wear curve.
