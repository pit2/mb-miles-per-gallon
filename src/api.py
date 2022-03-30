from flask import Flask, Response, request
import pandas as pd
import os
from src.predict import make_prediction

app = Flask(__name__)

training_data = pd.read_csv(os.path.join("data", "auto-mpg.csv"))


@app.route('/', methods=["GET"])
def main():
    return {"hello": "world"}


@app.route("/hello_world", methods=["GET"])
def hello_world():
    return "<p>Hello, World</p>"


@app.route("/training_data", methods=["GET"])
def get_training_data():
    return Response(training_data.to_json(), mimetype="application/json")


@app.route("/predict", methods=["GET"])
def predict_mpg():
    _ = request.args.get("ps")
    _ = request.args.get("zylinder")
    weight = request.args.get("gewicht")
    acc = request.args.get("beschleunigung")
    year = request.args.get("baujahr")

    return {"result": make_prediction(weight, acc, year)}
