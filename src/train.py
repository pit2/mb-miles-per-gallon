import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv("../data/auto-mpg.csv", sep=";")

df["gewicht2"] = np.square(df["gewicht"])
df["ps^2"] = np.square(df["ps"])
df["ps^3"] = np.power(df["ps"], 3)
df["beschleunigung^2"] = np.square(df["beschleunigung"])


# print(df.head())


X_train, X_test, y_train, y_test = model_selection.train_test_split(df.drop("mpg", axis=1),
                                                                    df["mpg"], train_size=0.7,
                                                                    random_state=1)

print("Fitting linear model...")
pipe = Pipeline([("scaler", MinMaxScaler(copy=False)),
                 ("linear_model", LinearRegression())])

pipe.fit(X_train, y_train)
y_predict = pipe.predict(X_test)

print(f"Model score: {pipe.score(X_test, y_test)}")
print(f"MSE: {mean_squared_error(y_test, y_predict)}")
coefficients = pipe.named_steps["linear_model"].coef_
print(f"Model coefficients: {coefficients}")
intercept = pipe.named_steps["linear_model"].intercept_
print(f"Intercept: {intercept}")

pickle.dump(pipe, open("../data/models/final_model", "wb"))
