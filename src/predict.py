"""Predict script

Implements a function to load a model to make a prediction on given input parameters.
"""

import pandas as pd
import pickle


def make_prediction(weight, acc, year):
    """Use the loaded model to make a mpg prediction returned as float."""
    pipe = pickle.load(open("data/models/final_model.pickle", "rb"))
    feature_vector = pd.DataFrame(
        data={"gewicht": weight, "beschleunigung": acc, "baujahr": year}, index=[0])
    predict = pipe.predict(feature_vector)
    return predict[0]
