from prometheus_client import start_http_server, Counter, Gauge
import time
import random

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


if __name__ == "__main__":
    start_http_server(8001)

    while True:
        REQUEST_COUNT.inc()
        PREDICTION_COUNT.inc()
        MODEL_ACCURACY.set(0.9649)
        MODEL_F1_SCORE.set(0.95)
        LATENCY.set(random.uniform(0.01, 0.20))

        time.sleep(5)