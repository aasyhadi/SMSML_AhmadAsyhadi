# SMSML Submission - Ahmad Asyhadi

Template proyek akhir SML: eksperimen, MLflow modelling, CI workflow, monitoring Prometheus, Grafana, dan inference.

## Cara lokal singkat
1. `cd Eksperimen_SML_AhmadAsyhadi && python preprocessing/automate_AhmadAsyhadi.py`
2. `mlflow ui --host 127.0.0.1 --port 5000`
3. `cd Membangun_model && python modelling.py && python modelling_tuning.py`
4. `cd ../Monitoring\ dan\ Logging && uvicorn prometheus_exporter:app --reload --port 8000`
5. Buka `http://127.0.0.1:8000/docs` dan `http://127.0.0.1:8000/metrics`.
