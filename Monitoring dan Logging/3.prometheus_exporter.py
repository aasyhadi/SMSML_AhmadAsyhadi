from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

HTTP_REQUESTS_TOTAL = Counter("http_requests_total", "Total HTTP Requests")
PREDICTION_REQUESTS_TOTAL = Counter("prediction_requests_total", "Total Prediction Requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction Latency Seconds")
LAST_PREDICTION_VALUE = Gauge("last_prediction_value", "Last Prediction Result")
SYSTEM_CPU_USAGE = Gauge("system_cpu_usage", "System CPU Usage Percentage")
SYSTEM_RAM_USAGE = Gauge("system_ram_usage", "System RAM Usage Percentage")


@app.route("/", methods=["GET"])
def home():
    HTTP_REQUESTS_TOTAL.inc()
    return jsonify({"status": "running"})


@app.route("/health", methods=["GET"])
def health():
    HTTP_REQUESTS_TOTAL.inc()
    return jsonify({"status": "healthy"})


@app.route("/metrics", methods=["GET"])
def metrics():
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=1))
    SYSTEM_RAM_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    HTTP_REQUESTS_TOTAL.inc()
    PREDICTION_REQUESTS_TOTAL.inc()

    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        result = response.json()

        prediction = result["predictions"][0]
        LAST_PREDICTION_VALUE.set(float(prediction))

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)