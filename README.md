# SMSML_AhmadAsyhadi

Submission Sistem Machine Learning Operations (MLOps) menggunakan dataset Breast Cancer Classification dengan implementasi:

* Eksperimen Machine Learning
* Model Development dan Experiment Tracking
* Workflow CI/CD menggunakan GitHub Actions
* Model Serving menggunakan FastAPI
* Monitoring menggunakan Prometheus
* Dashboard dan Alerting menggunakan Grafana

---

# Informasi Peserta 

**Nama:** Ahmad Asyhadi

**GitHub Repository:**
https://github.com/aasyhadi/SMSML_AhmadAsyhadi

---

# Struktur Repository

```text
SMSML_AhmadAsyhadi/
│
├── Eksperimen_SML_AhmadAsyhadi/
│   ├── preprocessing/
│   ├── namadataset_raw/
│   ├── .github/workflows/
│   ├── automate_AhmadAsyhadi.py
│   ├── Eksperimen_AhmadAsyhadi.ipynb
│   └── namadataset_preprocessing/
│
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   ├── artifacts/
│   ├── screenshot_dashboard.jpg
│   └── screenshot_artifak.jpg
│
├── Monitoring dan Logging/
│   ├── 1.bukti_serving.txt
│   ├── 2.prometheus.yml
│   ├── 3.prometheus_exporter.py
│   ├── 4.bukti monitoring Prometheus/
│   ├── 5.bukti monitoring Grafana/
│   ├── 6.bukti alerting Grafana/
│   └── 7.inference.py
│
├── Workflow-CI/
│   ├── .github/workflows/
│   │   └── ci-mlflow.yml
│   └── MLProject/
│
├── README.md
├── requirements.txt
└── Eksperimen_SML_AhmadAsyhadi.txt
```

---

# Dataset

Dataset yang digunakan adalah Breast Cancer Wisconsin Dataset.

Target klasifikasi:

* 0 = Malignant
* 1 = Benign

Dataset telah melalui tahapan:

* Data cleaning
* Feature selection
* Train-test split
* Data preprocessing

---

# Kriteria 1 - Eksperimen Machine Learning

Tahapan eksperimen meliputi:

1. Exploratory Data Analysis (EDA)
2. Data preprocessing
3. Training model
4. Evaluasi model
5. Otomatisasi workflow eksperimen

Output:

* Notebook eksperimen
* Dataset hasil preprocessing
* Workflow preprocessing otomatis

---

# Kriteria 2 - Membangun Model

Model yang digunakan:

* Random Forest Classifier

Framework:

* Scikit-Learn
* MLflow

Fitur yang diimplementasikan:

* Experiment Tracking
* Parameter Logging
* Metrics Logging
* Artifact Logging

Artifact yang dihasilkan:

* Classification Report
* Confusion Matrix
* Trained Model

---

# Kriteria 3 - Workflow CI

Workflow CI dibuat menggunakan GitHub Actions.

File:

```text
.github/workflows/ci-mlflow.yml
```

Proses yang dijalankan:

1. Checkout Repository
2. Setup Python Environment
3. Install Dependencies
4. Menjalankan MLflow Project
5. Verifikasi Training Pipeline

Status workflow dapat dilihat pada tab GitHub Actions.

---

# Kriteria 4 - Model Serving

Model disajikan menggunakan FastAPI.

Menjalankan aplikasi:

```bash
uvicorn app:app --reload
```

Endpoint:

```text
GET /
GET /health
POST /predict
```

Contoh:

```json
{
  "status": "running"
}
```

---

# Monitoring Menggunakan Prometheus

Menjalankan exporter:

```bash
python 3.prometheus_exporter.py
```

Prometheus:

```text
http://localhost:9090
```

Metrics yang dipantau:

* http_requests_total
* prediction_total
* last_prediction_value
* inference_latency_seconds
* system_cpu_usage
* system_ram_usage
* model_accuracy
* model_f1_score

---

# Dashboard Grafana

Grafana:

```text
http://localhost:3000
```

Dashboard:

```text
Dashboard-aasyhadi
```

Panel Monitoring:

* Total Request API
* Total Prediksi
* Last Prediction Value
* Penggunaan CPU
* Penggunaan RAM
* Latency Inference

---

# Alerting Grafana

Rule yang dibuat:

1. HTTP Requests High
2. CPU Usage High
3. Inference Latency High

Status alert dapat dipantau melalui:

```text
Alerting → Alert Rules
```

---

# Teknologi yang Digunakan

* Python 3.10+
* FastAPI
* Scikit-Learn
* MLflow
* GitHub Actions
* Prometheus
* Grafana
* Pandas
* NumPy

---

# Cara Menjalankan Project

## Install Dependency

```bash
pip install -r requirements.txt
```

## Jalankan FastAPI

```bash
uvicorn app:app --reload
```

## Jalankan Prometheus Exporter

```bash
python 3.prometheus_exporter.py
```

## Jalankan Prometheus

```bash
prometheus.exe --config.file=prometheus.yml
```

## Jalankan Grafana

```bash
grafana server
```

---

# Author

Ahmad Asyhadi

Submission Sistem Machine Learning Operations (MLOps) - Dicoding
