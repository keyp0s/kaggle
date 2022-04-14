# data analysis libraries
import pandas as pd
import numpy as np

# visualization libraries
import matplotlib.pyplot as plt

# ml framework
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

train = pd.read_csv("C:/repos/kaggle-comps/digit-recognizer/train.csv")
test = pd.read_csv("C:/repos/kaggle-comps/digit-recognizer/test.csv")

X = train.drop(columns = ['label'])
y = train.label

X_train, X_val, y_train ,y_val = train_test_split(X,y,test_size=0.2)

model = Sequential()

# need to reaserch what layers and activation functions to use
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=256,validation_data=(X_val, y_val))

#plotting
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'b', label='training loss')
plt.plot(epochs, val_loss,'r', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.legend()
plt.show()

y_test = model.predict(test)
y_test = np.array(pd.DataFrame(y_test).idxmax(axis=1))

submission = pd.read_csv('C:/repos/kaggle-comps/digit-recognizer/sample_submission.csv')
submission['Label']=y_test
submission.to_csv('submission1.csv',index=False)

print(submission.head(10))
