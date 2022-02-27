import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv("../data/auto-mpg.csv", sep=";")
df = df.drop("zylinder", axis=1)
df = df.drop("ps", axis=1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(df.drop("mpg", axis=1),
                                                                    df["mpg"], train_size=0.7)

print("Fitting linear model...")
pipe = Pipeline([("scaler", MinMaxScaler(copy=False)),
                 ("polnomial", PolynomialFeatures(
                     degree=(1, 3), include_bias=False)),
                 ("linear_model", Ridge(alpha=0.1))])

pipe.fit(X_train, y_train)
y_predict_train = pipe.predict(X_train)
y_predict_test = pipe.predict(X_test)


print(f"Model score: {pipe.score(X_test, y_test)}")
print(f"MSE on train set: {mean_squared_error(y_train, y_predict_train)}")
print(f"MSE on test set: {mean_squared_error(y_test, y_predict_test)}")

pickle.dump(pipe, open("../data/models/final_model", "wb"))
