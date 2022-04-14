# data analysis libraries
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt

# ml framework
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

train = pd.read_csv("C:/repos/kaggle-comps/digit-recognizer/train.csv")
test = pd.read_csv("C:/repos/kaggle-comps/digit-recognizer/test.csv")

x = train.drop(columns = ['label'])
y = train.label

x_train, x_test, y_train ,y_test = train_test_split(x,y,test_size=0.2)

model = Sequential()

# need to reaserch what layers and activation functions to use
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=32)