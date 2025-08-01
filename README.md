
# 🩺 VitalSynth API

VitalSynth delivers high-fidelity, HIPAA-safe synthetic ECG/PPG waveform data for AI development, testing, and simulation.

## 🔧 Features

- `/generate` endpoint (JSON-in, waveform-out)
- Differential privacy support (Opacus)
- Edge-case injection (AFib, dropout, motion artifacts)
- SDK and CLI usage
- Ready for GCP, Render, or local Docker

---

## 🚀 Local Usage

```bash
# Build and run locally
./launch.sh
```

Then go to `http://localhost:8000/docs` to access Swagger UI.

---

## 🔁 Endpoints

### `POST /generate`

```json
{
  "num_samples": 10,
  "edge_cases": true
}
```

Returns list of `[time, [ecg, ppg]]` sequences.

---

### `GET /metrics`

Returns training stats (epsilon, delta, MMD).

---

## 🧠 SDK Sample

```python
from vitalsynth_sdk import VitalSynthClient
client = VitalSynthClient()
samples = client.generate(100, edge_cases=True)
client.save_csv(samples, "waveform.csv")
```

---

## 📦 Deployment Options

### 🔹 Render.com

Use `render.yaml` to auto-deploy via Git integration.

### 🔹 Google Cloud

Use `cloudbuild.yaml` to build & push a container.

---

© 2025 VitalSynth
