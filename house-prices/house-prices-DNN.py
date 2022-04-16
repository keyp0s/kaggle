import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

train = pd.read_csv("C:/repos/kaggle-comps/house-prices/train.csv")
test = pd.read_csv("C:/repos/kaggle-comps/house-prices/test.csv")


#sns.histplot(train['SalePrice'])
#plt.show()
def clean_data(data):
    data = data.drop(["Id"], axis=1)
    for col in data.columns:
        if data[col].dtypes == object:
            data[col] = LabelEncoder().fit_transform(data[col])
        data.col = data[col].fillna(data[col].median(), inplace=True)
    return data

train = clean_data(train)
X_test = clean_data(test)

X = train.drop(['SalePrice'], axis=1)
y = train["SalePrice"]

X_train, X_val, y_train ,y_val = train_test_split(X,y,test_size=0.1)

model = Sequential()

# need to reaserch what layers and activation functions to use
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='relu'))

X_train = train.drop(['SalePrice'], axis=1)
y_train = train["SalePrice"]

model.compile(optimizer='adam',loss='mean_squared_logarithmic_error')

history = model.fit(X_train,y_train,epochs=256,validation_data=(X_val, y_val))

#plotting
loss = history.history['loss']
val_loss = history.history['val_loss']

start = int(len(loss)/10)

epochs = range(start+1, len(loss) + 1)
plt.plot(epochs, loss[start:], 'b', label='training loss')
plt.plot(epochs, val_loss[start:],'r', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.legend()
plt.show()

y_test = model.predict(X_test).flatten()

submit = pd.DataFrame({"Id":test["Id"], "SalePrice": y_test})
submit.to_csv("submit.csv", index=False)