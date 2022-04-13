# Data analysis libraries
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Kaggle Titanic - Machine Learning from Disaster Submission
# Training and testing data can be found here https://www.kaggle.com/competitions/titanic/data

train = pd.read_csv("C:/repos/kaggle-comps/titanic/train.csv")
test = pd.read_csv("C:/repos/kaggle-comps/titanic/test.csv")

sex = {"female": 0, "male": 1}
embarked = {"C": 0, "Q": 1,"S": 2, "U": 3}

def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    cols = ["SibSp","Parch","Fare","Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)

    data.Embarked.fillna("U", inplace=True)
    data = data.replace({"Sex":sex})
    data = data.replace({"Embarked":embarked})

    return data

train = clean(train)

predictors = train.drop(['Survived'], axis=1)
target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)

logModel = LogisticRegression()
logModel.fit(x_train, y_train)
print(logModel.score(x_train, y_train))