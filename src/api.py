from flask import Flask, Response
import pandas as pd
import os

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
