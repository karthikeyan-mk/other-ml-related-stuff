import pandas as pd
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Churn_Modelling.csv")
print(data.head(3))

print(data.shape)


print(data.describe())

x = data.iloc[:, 3:13].values
y = data.iloc[:, 13].values
print(x)
print(y)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x = LabelEncoder()
x[:, 1] = label_x.fit_transform(x[:, 1])
label_y = LabelEncoder()
x[:, 2] = label_y.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()

print(x)

x = x[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#festure scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import keras
from keras.layers import Dense
from keras.models import Sequential
classifier = Sequential()

#input layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to training set
classifier.fit(x_train, y_train, batch_size = 25, epochs = 500)

#predicting the result
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)