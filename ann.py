import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('german_credit.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_testM = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'sigmoid', input_dim = 20))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'sigmoid'))
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'sigmoid'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# second ann
classifier2 = Sequential()
classifier2.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 20))
classifier2.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))
classifier2.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
classifier2.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu'))
classifier2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier2.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# third ann
classifier3 = Sequential()
classifier3.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 20))
classifier3.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))
classifier3.add(Dense(output_dim = 10, init = 'uniform', activation = 'sigmoid'))
classifier3.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier3.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier3.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# fourth ann
classifier4 = Sequential()
classifier4.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 20))
classifier4.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))
classifier4.add(Dense(output_dim = 20, init = 'uniform', activation ='sigmoid' ))
classifier4.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier4.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier4.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# fifth ann
# fourth ann
classifier5 = Sequential()
classifier5.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 20))
classifier5.add(Dense(output_dim = 20, init = 'uniform', activation ='sigmoid' ))
classifier5.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier5.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier5.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


classifier6 = Sequential()
classifier6.add(Dense(output_dim = 10, init = 'uniform', activation = 'sigmoid', input_dim = 20))
classifier6.add(Dense(output_dim = 10, init = 'uniform', activation ='sigmoid' ))
classifier6.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier6.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier6.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

classifier7 = Sequential()
classifier7.add(Dense(output_dim = 20, init = 'uniform', activation = 'tanh', input_dim = 20))
classifier7.add(Dense(output_dim = 10, init = 'uniform', activation ='tanh' ))
classifier7.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier7.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier7.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

classifier8 = Sequential()
classifier8.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 20))
classifier8.add(Dense(output_dim = 10, init = 'uniform', activation ='tanh' ))
classifier8.add(Dense(output_dim = 10, init = 'uniform', activation ='sigmoid' ))
classifier8.add(Dense(output_dim = 10, init = 'uniform', activation ='sigmoid' ))
classifier8.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier8.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier8.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

classifier9 = Sequential()
classifier9.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 20))
classifier9.add(Dense(output_dim = 10, init = 'uniform', activation ='sigmoid'))
classifier9.add(Dense(output_dim = 10, init = 'uniform', activation ='sigmoid' ))
classifier9.add(Dense(output_dim = 10, init = 'uniform', activation ='sigmoid' ))
classifier9.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier9.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier9.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)




hold=[]
y_pred1 = classifier.predict(X_test)
y_pred1 = (y_pred1 > 0.5)
y_pred2 = classifier2.predict(X_test)
y_pred2 = (y_pred2 > 0.5)
y_pred3 = classifier3.predict(X_test)
y_pred3 = (y_pred3 > 0.5)
y_pred4 = classifier4.predict(X_test)
y_pred4 = (y_pred4 > 0.5)
y_pred5 = classifier5.predict(X_test)
y_pred5 = (y_pred5 > 0.5)
y_pred6 = classifier6.predict(X_test)
y_pred6 = (y_pred6 > 0.5)
y_pred7 = classifier7.predict(X_test)
y_pred7 = (y_pred7 > 0.5)
y_pred8 = classifier8.predict(X_test)
y_pred8 = (y_pred8 > 0.5)
y_pred9 = classifier9.predict(X_test)
y_pred9 = (y_pred9 > 0.5)
error1= np.mean( y_pred1  != y_testM)
hold.append(error1)  
error2= np.mean( y_pred2  != y_testM)
hold.append(error2) 
error3= np.mean( y_pred3  != y_testM)
hold.append(error3) 
error4= np.mean( y_pred4  != y_testM)
hold.append(error4)  
error5= np.mean( y_pred5  != y_testM)
hold.append(error5) 
error6= np.mean( y_pred6  != y_testM)
hold.append(error6) 
error7= np.mean( y_pred7  != y_testM)
hold.append(error7) 
error8= np.mean( y_pred8  != y_testM)
hold.append(error8) 
error9= np.mean( y_pred9  != y_testM)
hold.append(error9) 