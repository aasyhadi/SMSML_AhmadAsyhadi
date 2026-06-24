from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}