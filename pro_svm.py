# -*- coding: utf-8 -*-
"""
Created on

@author: Erinc
"""



# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.preprocessing import StandardScaler
# Importing the dataset
sc=StandardScaler()
dataset = pd.read_csv('german_credit.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
hold1=[] 
hold2=[]
hold3=[]
hold4=[]
hold5=[]
hold6=[]
hold7=[]
hold8=[]
hold9=[]
hold111=[]
hold112=[]
hold113=[]
hold114=[]
hold115=[]
hold116=[]
hold117=[]
hold118=[]
# split data

from sklearn.cross_validation import train_test_split
for i in range(1,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = i)
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    # svm
    #for i in range(20):
    classifier=SVC(kernel='poly',degree=6)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    error= np.mean( y_test != y_pred)
    hold1.append(error)
    y_pred11=classifier.predict(X_train)
    error11= np.mean( y_train != y_pred11)
    hold111.append(error11)
    #
    classifier2=SVC(kernel='linear')
    classifier2.fit(X_train,y_train)
    y_pred2=classifier2.predict(X_test)
    error2= np.mean( y_test != y_pred2)
    hold2.append(error2)
    y_pred112=classifier2.predict(X_train)
    error112= np.mean( y_train != y_pred112)
    hold112.append(error112)
    
    
    classifier3=SVC(kernel='rbf')
    classifier3.fit(X_train,y_train)
    y_pred3=classifier3.predict(X_test)
    error3= np.mean( y_test != y_pred3)
    hold3.append(error3)
    y_pred113=classifier3.predict(X_train)
    error113= np.mean( y_train != y_pred113)
    hold113.append(error113)
    
    
    classifier4=SVC(kernel='sigmoid')
    classifier4.fit(X_train,y_train)
    y_pred4=classifier4.predict(X_test)
    error4= np.mean( y_test != y_pred4)
    hold4.append(error4) 
    y_pred114=classifier4.predict(X_train)
    error114= np.mean( y_train != y_pred114)
    hold114.append(error114)
    
##
    classifier5=NuSVC(nu=0.3,kernel='linear')#,gamma=1,decision_function_shape='ovr')
    classifier5.fit(X_train,y_train)
    y_pred5=classifier5.predict(X_test)
    error5 = np.mean( y_test != y_pred5)
    hold5.append(error5)
    y_pred115=classifier5.predict(X_train)
    error115= np.mean( y_train != y_pred115)
    hold115.append(error115)
    
    classifier6=NuSVC(nu=0.3,kernel='poly')#,gamma=1,degree=6,decision_function_shape='ovr')
    classifier6.fit(X_train,y_train)
    y_pred6=classifier6.predict(X_test)
    error6= np.mean( y_test != y_pred6)
    hold6.append(error6)
    y_pred116=classifier6.predict(X_train)
    error116= np.mean( y_train != y_pred116)
    hold116.append(error116)
    
    
    classifier7=NuSVC(nu=0.3,kernel='rbf')#,gamma=1,decision_function_shape='ovr')
    classifier7.fit(X_train,y_train)
    y_pred7=classifier7.predict(X_test)
    error7= np.mean( y_test != y_pred7)
    hold7.append(error7)
    y_pred117=classifier7.predict(X_train)
    error117= np.mean( y_train != y_pred117)
    hold117.append(error117)  
    
    classifier8=NuSVC(nu=0.3,kernel='sigmoid')#,gamma=1,decision_function_shape='ovr')
    classifier8.fit(X_train,y_train)
    y_pred8=classifier8.predict(X_test)
    error8= np.mean( y_test != y_pred8)
    hold8.append(error8)
    y_pred118=classifier8.predict(X_train)
    error118= np.mean( y_train != y_pred118)
    hold118.append(error118) 
    
    classifier9=LinearSVC(C=0.9)
    classifier9.fit(X_train,y_train)
    y_pred9=classifier9.predict(X_test)
    error9 = np.mean( y_train != y_pred9)
    hold9.append(error9)


x= np.arange(1,100)
testMeanerror=[]
traingMeanerror=[]
## explained variance
meantest1=np.mean(y)
y=np.array(hold1)
y111=np.array(hold111)
meantest111=np.mean(y111)
traingMeanerror.append(meantest111)
testMeanerror.append(meantest1)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(x,y)

#plt.yticks(np.arange(y4.min(), y4.max(),0.1))
plt.xlabel('Number of Trial')
plt.ylabel('Error')
plt.title('SVC Poly')
plt.savefig('SVC Poly.jpg')
#Image.open('PCA Variance.png').save('PCA Variance.jpg','JPEG')
y2=np.array(hold2)
meantest2=np.mean(y2)
testMeanerror.append(meantest2)
y112=np.array(hold112)
meantest112=np.mean(y112)
traingMeanerror.append(meantest112)
import matplotlib.pyplot as plt
plt.figure(2)
plt.plot(x,y2)

#plt.yticks(np.arange(y.min()-0.005, y.max()+0.05, 0.05))
plt.xlabel('Number of Trial')
plt.ylabel('Error')
plt.title('SVC Linear')
plt.savefig('SVC Linear.jpg')
#Image.open('PCA Logistic Regression.png').save('PCA Logistic Regression.jpg','JPEG')
#
plt.figure(3)
y3=np.array(hold3)
meantest3=np.mean(y3)
testMeanerror.append(meantest3)
y113=np.array(hold113)
meantest113=np.mean(y113)
traingMeanerror.append(meantest113)
plt.plot(x,y3)

plt.xlabel('Number of Trial')
plt.ylabel('Error')
plt.title('SVC rbf')
plt.savefig('SVC rbf.jpg')
#
plt.figure(4)
y4=np.array(hold4)
meantest4=np.mean(y4)
testMeanerror.append(meantest4)
y114=np.array(hold114)
meantest114=np.mean(y114)
traingMeanerror.append(meantest114)
plt.plot(x,y4)
plt.xlabel('Number of Trial')

plt.ylabel('Error')
plt.title('SVC Sigmoid')
plt.savefig('SVC Sigmoid.jpg')

plt.figure(5)
y5=np.array(hold5)
meantest5=np.mean(y5)
testMeanerror.append(meantest5)
y115=np.array(hold115)
meantest115=np.mean(y115)
traingMeanerror.append(meantest115)
plt.plot(x,y5)
plt.xlabel('Number of Trial')

plt.ylabel('Error')
plt.title('NuSVC Nu=0.3 Linear')
plt.savefig('NuSVC Nu=0.3 Linear.jpg')

plt.figure(6)
y6=np.array(hold6)
meantest6=np.mean(y6)
testMeanerror.append(meantest6)
y116=np.array(hold116)
meantest116=np.mean(y116)
traingMeanerror.append(meantest116)
plt.plot(x,y6)
plt.xlabel('Number of Trial')

plt.ylabel('Error')
plt.title('NuSVC Nu=0.3 Poly')
plt.savefig('NuSVC Nu=0.3 poly.jpg')

plt.figure(7)
y7=np.array(hold7)
meantest7=np.mean(y7)
testMeanerror.append(meantest7)
y117=np.array(hold117)
meantest117=np.mean(y117)
traingMeanerror.append(meantest117)
plt.plot(x,y7)
plt.xlabel('Number of Trial')

plt.ylabel('Error')
plt.title('NuSVC Nu=0.3 Rbf')
plt.savefig('NuSVC Nu=0.3 Rbf.jpg')


plt.figure(8)
y8=np.array(hold8)
meantest8=np.mean(y8)
testMeanerror.append(meantest8)
y118=np.array(hold118)
meantest118=np.mean(y118)
traingMeanerror.append(meantest118)
plt.plot(x,y8)
plt.xlabel('Number of Trial')

plt.ylabel('Error')
plt.title('NuSVC Nu=0.3 sigmoid')
plt.savefig('NuSVC Nu=0.3 sigmoid.jpg')
np.save('testmeanErrror',testMeanerror)
np.save('traingmeanErrror',traingMeanerror)