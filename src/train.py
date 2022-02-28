"""Train script for data preparation and polynomial regression

First, this script prepares the data by dropping the zylinder and ps feature columns as they have
been found to have little predictive value. The remaining features are then Min-Max-normalized
to have range [0,1]. A quadratic polynomial model is then trained with ridge regression and saved.
"""

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import pickle

DROP_COLS = ("zylinder", "ps")

df = pd.read_csv("../data/auto-mpg.csv", sep=";")


def drop(columns, in_df):
    """Return modified dataframe, i.e. one where the specified columns have been dropped.

    Keyword arguments:
    columns -- a list of strings specifying the columns to be dropped
    in_df -- a pandas dataframe from which the columns are dropped
    """
    for col in columns:
        in_df = in_df.drop(col, axis=1)
    return in_df


df = drop(("zylinder", "ps"), df)

X_train, X_test, y_train, y_test = model_selection.train_test_split(df.drop("mpg", axis=1),
                                                                    df["mpg"], train_size=0.7)

print("Fitting linear model...")
pipe = Pipeline([("scaler", MinMaxScaler(copy=False)),
                 ("polnomial", PolynomialFeatures(
                     degree=(1, 2), include_bias=False)),
                 ("linear_model", Ridge(alpha=0.1))])

pipe.fit(X_train, y_train)
y_predict_train = pipe.predict(X_train)
y_predict_test = pipe.predict(X_test)

print(f"Model score: {pipe.score(X_test, y_test)}")
print(
    f"RMSE on train set: {np.sqrt(mean_squared_error(y_train, y_predict_train))}")
print(
    f"RMSE on test set: {np.sqrt(mean_squared_error(y_test, y_predict_test))}")

pickle.dump(pipe, open("../data/models/final_model", "wb"))
