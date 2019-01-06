# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
# Importing the dataset
dataset = pd.read_csv('german_credit.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
sc = StandardScaler()
classifier = GaussianNB()
hold=[]
hold2=[]
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred2 = classifier.predict(X_train)
    error = np.mean( y_test != y_pred)
    hold.append(error)
    error2 = np.mean( y_train != y_pred2)
    hold2.append(error2)
    
x= np.arange(100)
#from numpy import array
y=np.array(hold)
y2=np.array(hold2)
meaz=np.mean(y)
meazs=np.mean(y2)
cvar=np.var(y)
carea=np.var(y2)
np.save('travar',y2)
np.save('testvar',y)
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(x,y)
plt.xlabel('Iteration Number')
plt.ylabel('Error%')
plt.title('Naive Bayes Test Set')
plt.savefig('naivebayes.jpg')
import matplotlib.pyplot as plt
plt.figure(2)
plt.plot(x,y2)
plt.xlabel('Iteration Number')
plt.ylabel('Error%')
plt.title('Naive Bayes Training Set')
plt.savefig('naivebayestr.jpg')
#Image.open('naivebayes.png').save('naivebayes.jpg','JPEG')
