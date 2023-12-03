from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##--------------------------------origin-------------------------------------------##
dataset = loadtxt('/Users/mingjian/PycharmProjects/Find_env/Pima.csv', delimiter=',')
X = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history1 = model.fit(X, y, epochs=200, batch_size=10)
print(history1.history.keys())

##--------------------------------optimal------------------------------------------##
dataset = loadtxt('/Users/mingjian/PycharmProjects/Find_env/Pima_opti.csv', delimiter=',')
X = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = model.fit(X, y, epochs=200, batch_size=10)
print(history2.history.keys())

##----------------------------------request---------------------------------------##
dataset = loadtxt('/Users/mingjian/PycharmProjects/Find_env/Pima_worst.csv', delimiter=',')
X = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history3 = model.fit(X, y, epochs=200, batch_size=10)
print(history3.history.keys())



plt.plot(history1.history['accuracy'], 'r--', label='Original Model (Service Provider Dominated)')
plt.plot(history2.history['accuracy'], '.', color='blue', label='Our Optimal Solution')
plt.plot(history3.history['accuracy'], '^', color='green', label='All Customers Requests (Customers Dominated)')
plt.title('Model Accuracy of Pima Dataset')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

#plt.plot(history.history['loss'], 'r--', label='100% of Pima')
#plt.title('Model loss of Pima Dataset')
#plt.ylabel('Loss')
#plt.xlabel('Epochs')
#plt.legend()
#plt.show()




