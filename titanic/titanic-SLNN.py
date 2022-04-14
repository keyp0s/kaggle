import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("C:/repos/kaggle-comps/titanic/train.csv")
test = pd.read_csv("C:/repos/kaggle-comps/titanic/test.csv")

sex = {"female": 0, "male": 1}
embarked = {"C": 0, "Q": 1,"S": 2, "U": 3}

def clean_data(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    cols = ["SibSp","Parch","Fare","Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)

    data.Embarked.fillna("U", inplace=True)
    data = data.replace({"Sex":sex})
    data = data.replace({"Embarked":embarked})

    return data

def train_model(data):
    X_train = train.drop(['Survived'], axis=1)
    y_train = train["Survived"]

    model = LogisticRegression()

    return model.fit(X_train, y_train)

train = clean_data(train)
test_train = clean_data(test)

model = train_model(train)
y_test = model.predict(test_train)

submit = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived": y_test})
submit.to_csv("submit.csv", index=False)