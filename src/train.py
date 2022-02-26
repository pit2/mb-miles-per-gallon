import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
# import pickle

df = pd.read_csv("../data/auto-mpg.csv", sep=";")

#df["gewicht2"] = np.square(df["gewicht"])
#df["gewicht^3"] = np.power(df["gewicht"], 3)
#df["ps^2"] = np.square(df["ps"])
#df["ps^3"] = np.power(df["ps"], 3)
# df["beschleunigung^2"] = np.square(df["beschleunigung"])
# df["bschleunigung"]


print(df.head())


X_train, X_test, y_train, y_test = model_selection.train_test_split(df.drop("mpg", axis=1),
                                                                    df["mpg"], train_size=0.3)

normalisation = {}


def normalise_col(data, min_, max_):
    if min_ is None:
        min_ = np.min(data)
    if max_ is None:
        max_ = np.max(data)
    return data - min_ / (max_ - min_), (min_, max_)


def normalise(features, target, target_str, with_min_max=False):
    for col in features.columns:
        if with_min_max:
            min_ = normalisation[col][0]
            max_ = normalisation[col][1]
        else:
            min_, max_ = None, None
        features[col], normalisation[col] = normalise_col(
            features[col], min_, max_)
    target, normalisation[target_str] = normalise_col(target, min_, max_)


# normalise(X_train, y_train, "mpg")
X_train["gewicht^2"] = np.square(X_train["gewicht"])
X_train["ps^2"] = np.square(X_train["ps"])
X_test["gewicht^2"] = np.square(X_test["gewicht"])
X_test["ps^2"] = np.square(X_test["ps"])

print("Beginning fitting linear model...")
model = Pipeline([
                 ("linear_model", LinearRegression())])
model.fit(X_train, y_train)

# normalise(X_test, y_test, "mpg", True)


print(f"Model score: {model.score(X_test, y_test)}")
coefficients = model.named_steps["linear_model"].coef_
print(f"Model coefficients: {coefficients}")
