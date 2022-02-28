"""Predict script

Prompt the user for input and print a prediction in the terminal for the miles-per-gallon as
predicted by the loaded model.
"""

import pandas as pd
import pickle

pipe = pickle.load(open("../data/models/final_model", "rb"))

print("Enter values for features to make a prediction.")
weight = int(input("Weight: "))
acc = float(input("Acceleration: "))
year = int(input("Construction year: "))
# 3433, 12.0, 70
feature_vector = pd.DataFrame(data={"gewicht": weight, "beschleunigung": acc, "baujahr": year},
                              index=[0])
predict = pipe.predict(feature_vector)
print(f"Predicted miles per gallon: {predict}")
