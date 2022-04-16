#submission for https://kaggle.com/competitions/titanic

import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("C:/repos/kaggle-comps/titanic/train.csv")
test = pd.read_csv("C:/repos/kaggle-comps/titanic/test.csv")

def clean_data(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    cols = ["SibSp","Parch","Fare","Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)

    data.Embarked.fillna("C", inplace=True)
    data = data.replace({"Sex":{"female": 0, "male": 1}})
    data = data.replace({"Embarked":{"C": 0, "Q": 1,"S": 2}})

    return data

train = clean_data(train)
X_test = clean_data(test)

X_train = train.drop(['Survived'], axis=1)
y_train = train["Survived"]

model = LogisticRegression()
model.fit(X_train, y_train)

y_test = model.predict(X_test)

submit = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived": y_test})
submit.to_csv("submit.csv", index=False)