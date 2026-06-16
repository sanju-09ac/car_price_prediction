import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("car data.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df["Car_Age"] = 2026 - df["Year"]

df.drop(["Year"], axis=1, inplace=True)

df.drop("Car_Name", axis=1, inplace=True)

df = pd.get_dummies(
    df,
    columns=["Fuel_Type",
             "Selling_type",
             "Transmission"],
    drop_first=True
)
X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))

print("R2 Score:", r2_score(y_test, pred))

plt.scatter(y_test, pred)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted")

plt.show()
