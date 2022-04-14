# data analysis libraries
import pandas as pd
import numpy as np

# visualization libraries
import matplotlib.pyplot as plt

# ml framework
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

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

X = train.drop(['Survived'], axis=1)
y = train["Survived"]

X_train, X_val, y_train ,y_val = train_test_split(X,y,test_size=0.2)

model = Sequential()

# need to reaserch what layers and activation functions to use
model.add(Dense(64, activation='relu', input_shape=(7,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='RMSprop',loss='binary_crossentropy',metrics=['binary_accuracy'])

# overfitting issues
history = model.fit(X_train,y_train,epochs=2048,validation_data=(X_val, y_val))

#plotting
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'b', label='training loss')
plt.plot(epochs, val_loss,'r', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.legend()
plt.show()

y_predict = ((model.predict(test_train) > 0.5)*1).flatten()
submit = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived": y_predict})

# line below saves output to csv
submit.to_csv("submit.csv", index=False)
print(submit.head(10))