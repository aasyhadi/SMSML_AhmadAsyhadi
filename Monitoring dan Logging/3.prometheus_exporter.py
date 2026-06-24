from prometheus_client import start_http_server, Counter, Gauge
import time
import random
import psutil

# Business Metrics
REQUEST_COUNT = Counter(
    "inference_request_total",
    "Total jumlah request inference"
)

PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total jumlah prediksi"
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Akurasi model terakhir"
)

MODEL_F1_SCORE = Gauge(
    "model_f1_score",
    "F1 score model terakhir"
)

LATENCY = Gauge(
    "inference_latency_seconds",
    "Latency inference dalam detik"
)

LAST_PREDICTION = Gauge(
    "last_prediction_value",
    "Nilai prediksi terakhir"
)

# System Metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests"
)

SYSTEM_CPU_USAGE = Gauge(
    "system_cpu_usage",
    "CPU usage percentage"
)

SYSTEM_RAM_USAGE = Gauge(
    "system_ram_usage",
    "RAM usage percentage"
)

if __name__ == "__main__":
    start_http_server(8001)

    while True:
        # Simulasi request
        REQUEST_COUNT.inc()
        PREDICTION_COUNT.inc()
        HTTP_REQUESTS_TOTAL.inc()

        # Model metrics
        MODEL_ACCURACY.set(0.9649)
        MODEL_F1_SCORE.set(0.95)
        LATENCY.set(random.uniform(0.01, 0.20))

        prediction = random.choice([0, 1])
        LAST_PREDICTION.set(prediction)

        # System metrics
        SYSTEM_CPU_USAGE.set(psutil.cpu_percent())

        SYSTEM_RAM_USAGE.set(
            psutil.virtual_memory().percent
        )

        time.sleep(5)