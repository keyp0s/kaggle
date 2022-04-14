# data analysis libraries
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt

# ml framework
from sklearn.linear_model import LogisticRegression

# kaggle titanic - machine learning from disaster submission
# training and testing data can be found here https://www.kaggle.com/competitions/titanic/data
# my first kaggle submission, at time of submitting scores 0.76794% accuracy (much room for improvement)

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
test_train = clean(test)

x_train = train.drop(['Survived'], axis=1)
y_train = train["Survived"]

logModel = LogisticRegression()
logModel.fit(x_train, y_train)
accuracy = (logModel.score(x_train,y_train))
print (accuracy)

y_pred = logModel.predict(test_train)
print(y_pred.shape)

submit = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived": y_pred})

# line below saves output to csv
# submit.to_csv("submit.csv", index=False)
print(submit.head(10))