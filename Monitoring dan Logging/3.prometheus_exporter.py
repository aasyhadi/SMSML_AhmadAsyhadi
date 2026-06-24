from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests"
)

PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency seconds"
)

LAST_PREDICTION = Gauge(
    "last_prediction_value",
    "Last prediction result"
)

CPU_USAGE = Gauge(
    "system_cpu_usage",
    "CPU Usage Percentage"
)

RAM_USAGE = Gauge(
    "system_ram_usage",
    "RAM Usage Percentage"
)


@app.route("/", methods=["GET"])
def home():
    REQUEST_COUNT.inc()
    return jsonify({"status": "running"})


@app.route("/health", methods=["GET"])
def health():
    REQUEST_COUNT.inc()
    return jsonify({"status": "healthy"})


@app.route("/metrics", methods=["GET"])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    REQUEST_COUNT.inc()
    PREDICTION_COUNT.inc()

    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time

        REQUEST_LATENCY.observe(duration)

        result = response.json()

        try:
            prediction = result["predictions"][0]
            LAST_PREDICTION.set(float(prediction))
        except Exception:
            LAST_PREDICTION.set(0)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)