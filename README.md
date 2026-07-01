# Aditya-L1 Solar Flare Early Warning System
### ML-Powered Telemetry Processing & Forecast Dashboard (Ground Segment)

An end-to-end, production-ready Ground Segment processor and visualization dashboard designed to ingest, clean, and analyze real-time solar X-ray telemetry from the **SoLEXS** (Soft X-Ray) and **HEL1OS** (Hard X-Ray) instruments onboard ISRO's **Aditya-L1** observatory.

The system utilizes custom deep learning models (Autoencoders, 1D-CNNs, and BiLSTMs) to perform signal denoising, imminent precursor nowcasting, and classification/time-to-peak forecasting of solar flare events.

---

## 🚀 Key Features

*   **Denoising Autoencoder**: Filters raw counts telemetry, removing instrumental jitter and sensor noise.
*   **1D-CNN Nowcaster**: Predicts imminent solar flare probabilities at a 1-second cadence (Precursor Alert threshold: 44%).
*   **BiLSTM Forecaster**: Performs multi-task classification (GOES Flare Class: B, C, M, X) and regression (Estimated Time-to-Peak in hours) with a 15-minute lead time.
*   **Live Simulation Broadcaster**: Streams simulated CCSDS telemetry packets from an infinite synthetic generator via WebSockets (cycling through Quiet Sun, C-class Precursor, Peak, and Exponential Decay phases).
*   **Global Alerts Center & Popups**: Non-intrusive floating toasts in the bottom-right and a persistent bell-icon dropdown history panel in the navbar with 25-second duplicate suppression.
*   **Interactive Modal Visualizations**: Fullscreen graph modal expansions featuring Zoom (visible window size adjustment), Scroll (horizontal panning back in time), Signal Isolation, and CSV segment exports.
*   **Bulk Telemetry Exporter**: Upload raw historical CSV records to align timestamps, run ML inference, and export resampled datasets.
*   **Dual Themes**: Premium custom Space Dark Mode and high-contrast Light Mode toggles.

---

## 📂 Project Architecture

```
Solar_Flare_Detection_System/
├── backend/                  # FastAPI Application
│   ├── main.py               # API endpoints & WebSocket simulation loop
│   ├── inference.py          # TF/Keras model loader & inference pipeline
│   ├── schemas.py            # Pydantic validation schemas
│   └── requirements.txt      # Backend Python dependencies
│
├── dashboard/                # Next.js Frontend Application
│   ├── src/
│   │   ├── app/              # Routes (/, /simulation, /exporter)
│   │   ├── components/       # HeaderBar, AlertNotifications, AlertBanner
│   │   ├── context/          # TelemetryContext, ThemeContext
│   │   └── hooks/            # useTelemetrySimulator
│   ├── package.json          # Frontend Node dependencies
│   └── tailwind.config.ts    # Styling configurations
│
└── ml/                       # Machine Learning Codebase
    ├── models/               # Autoencoder, CNN, BiLSTM architectures
    ├── data/                 # Data cleaners, augmentations, and loaders
    └── training/             # PyTorch model training pipelines
```

---

## 🛠️ Installation & Setup

### Prerequisites
*   **Python 3.10+** (with `venv` support)
*   **Node.js 18+** & **npm**

---

### 1. Run the FastAPI Backend

1. Navigate to the project root directory:
   ```bash
   cd Solar_Flare_system
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install the backend Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. Start the FastAPI server using Uvicorn:
   ```bash
   uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *The API will be available at `${process.env.NEXT_PUBLIC_API_URL}` (docs at `/docs`).*

---

### 2. Run the Next.js Dashboard

1. Open a new terminal window and navigate to the dashboard directory:
   ```bash
   cd Solar_Flare_system/dashboard
   ```

2. Install the Node.js packages:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```
   *The dashboard will be accessible locally at `http://localhost:3000`.*

---

## 📋 Data Formats

### REST Endpoints
*   **`POST /predict/live`**: Ingests a single telemetry point `{`sxr_raw": float, "hxr_raw": float}` and returns denoised counts, alert levels, nowcast probabilities, and active forecasts.
*   **`POST /preprocess-and-clean`**: Uploads bulk telemetry CSV files, aligns timestamps, runs inference, and generates downloadable datasets.

### WebSocket (`ws://127.0.0.1:8000/ws/simulation`)
Streams a 1Hz telemetry packet log formatted as:
```json
{
  "tickCount": 1835,
  "sxr_raw": 3.69,
  "hxr_raw": 1.85,
  "sxr_clean": 3.69,
  "hxr_clean": 1.85,
  "nowcastProb": 0.04,
  "forecastClass": "B2.1",
  "forecastProb": 0.067,
  "forecastTimeToPeak": "12h 0m",
  "alertLevel": "normal"
}
```

---

## 🛡️ License
Designed and developed for ground segment space weather warning systems. Powered by TensorFlow and Next.js.
