# NeuroAegis

NeuroAegis is an end-to-end Machine Learning platform and Web Application designed for the detection and prediction of epileptic seizures from EEG (Electroencephalography) signals. 

By combining advanced signal processing, machine learning (LightGBM), and Explainable AI (SHAP), NeuroAegis provides clinicians and researchers with an intuitive interface to upload EEG data, receive rapid seizure predictions, and understand the exact features driving the model's decisions.

## 🌟 Key Features

- **Multi-Format EEG Uploads**: Seamlessly upload `.edf`, `.csv`, or `.txt` EEG files directly through the web interface.
- **Advanced Signal Processing**: Automatically extracts complex features across multiple domains:
  - **Time Domain**: Statistical moments, Hjorth parameters, Zero Crossing Rate, Line Length.
  - **Frequency Domain**: Power Spectral Density, Bandpowers (Delta, Theta, Alpha, Beta, Gamma).
  - **Wavelet Domain**: Discrete Wavelet Transform (DWT) energies and entropies.
- **Robust Machine Learning**: Powered by LightGBM models trained on renowned datasets (Bonn University and CHB-MIT).
- **Explainable AI (XAI)**: Integrated SHAP (SHapley Additive exPlanations) values to visualize which specific EEG features most strongly indicate seizure activity, fostering clinical trust.
- **Modern UI**: A beautifully designed, highly responsive dashboard built with React, TailwindCSS, and Framer Motion.

## 🏗️ Architecture

NeuroAegis is structured as a monorepo containing a full-stack application:

- **`apps/web/` (Frontend)**: 
  - React + TypeScript
  - Vite for fast bundling
  - TailwindCSS for styling
  - Framer Motion & Recharts for interactive animations and data visualization.

- **`apps/api/` (Backend)**:
  - FastAPI (Python)
  - Scikit-learn, LightGBM, and SHAP for the machine learning pipeline
  - PyWavelets & SciPy for signal processing
  - SQLite & SQLAlchemy for job tracking

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.11 or 3.12 recommended for scientific computing dependencies)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/NeuroAegis.git
cd NeuroAegis
```

### 2. Set up the Backend (FastAPI)
```bash
cd apps/api

# Create a virtual environment
python3.12 -m venv ../../.venv
source ../../.venv/bin/activate  # On Windows use `..\..\.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API will be available at `http://localhost:8000`.

### 3. Set up the Frontend (React/Vite)
Open a new terminal window:
```bash
cd apps/web

# Install dependencies
npm install

# Start the development server
npm run dev
```
The frontend will be available at `http://localhost:5173`.

### 4. Running with Docker (Alternative)
You can also spin up the entire stack using Docker Compose:
```bash
docker-compose up --build
```

## 🧠 Datasets & Models
The pipelines support predictive modeling based on:
1. **Bonn University Dataset**: A standard benchmark dataset for EEG seizure detection, containing single-channel EEG segments.
2. **CHB-MIT Scalp EEG Database**: A comprehensive pediatric dataset containing continuous multi-channel scalp EEG recordings.

Pre-trained `.pkl` models (LightGBM and Random Forest) are utilized by the backend to serve inference requests.

## 📄 License
This project is licensed under the terms specified in the `LICENSE` file.
