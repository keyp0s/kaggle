import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

train = pd.read_csv("C:/repos/kaggle-comps/house-prices/train.csv")
test = pd.read_csv("C:/repos/kaggle-comps/house-prices/test.csv")

def clean_data(data):
    for col in data.columns:
        if data[col].dtypes == object:
            data[col] = LabelEncoder().fit_transform(data[col])
        data.col = data[col].fillna(data[col].median(), inplace=True)
    return data

train = clean_data(train)
X_test = clean_data(test)

X_train = train.drop(['SalePrice'], axis=1)
y_train = train["SalePrice"]

model = LinearRegression()
model.fit(X_train, y_train)

y_test = model.predict(X_test)

submit = pd.DataFrame({"Id":test["Id"], "SalePrice": y_test})
submit.to_csv("submit.csv", index=False)