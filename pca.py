import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from PIL import Image
# Importing the dataset
dataset = pd.read_csv('german_credit.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
sc = StandardScaler()
regressor = LinearRegression()
lin_reg_2 = LinearRegression()

# Splitting the dataset into the Training set and Test set

hold=[]
hold2=[]
hold3=[]
exp_var=[]
exp_var2=[]
X_trainM, X_testM, y_trainM, y_testM = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_trainM = sc.fit_transform(X_trainM)
X_testM = sc.transform(X_testM)
classifier = LogisticRegression(random_state = 0)
for i in range(1,21):
    pca = PCA(n_components = i)
    X_train = pca.fit_transform(X_trainM)
    X_test = pca.transform(X_testM)  
    explained_variance = pca.explained_variance_ratio_
    exp_var32=sum(explained_variance)
    exp_var2.append(exp_var32)
    exp_var.append(explained_variance)
    # linear Regression
    regressor.fit(X_train, y_trainM)
    y_pred2= regressor.predict(X_train)
    y_pred2 = (y_pred2 > 0.5)
    error2 = np.mean( y_trainM != y_pred2)
    hold2.append(error2)
    # logistic regresion
    classifier.fit(X_train, y_trainM)
    y_pred = classifier.predict(X_train)
    error = np.mean( y_trainM != y_pred)
    hold.append(error)
    # polynamial regression   
    poly_reg = PolynomialFeatures(degree = 6)
    X_poly = poly_reg.fit_transform(X_train)
    lin_reg_2.fit(X_poly, y_trainM)
    y_pred3= lin_reg_2.predict(poly_reg.fit_transform(X_train))#lin_reg_2.predict(X_test)
    y_pred3 = (y_pred3 > 0.5)
    error3 = np.mean( y_trainM != y_pred3)
    hold3.append(error3)
    del X_train
    del X_test
    del explained_variance


x= np.arange(1,21)
## explained variance
y4=np.array(exp_var2)
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(x,y4)
plt.xticks(np.arange(x.min(), x.max()+1, 1))
#plt.yticks(np.arange(y4.min(), y4.max(),0.1))
plt.xlabel('Number of Component')
plt.ylabel('Explained Variance')
plt.title('PCA Variance')
plt.savefig('PCA Variance.jpg')
#Image.open('PCA Variance.png').save('PCA Variance.jpg','JPEG')
y=np.array(hold)
import matplotlib.pyplot as plt
plt.figure(2)
plt.plot(x,y)
plt.xticks(np.arange(x.min(), x.max(), 1))
#plt.yticks(np.arange(y.min()-0.005, y.max()+0.05, 0.05))
plt.xlabel('Number of Component')
plt.ylabel('Error%')
plt.title('Logistic Regression Training Set')
plt.savefig('PCA Logistic Regression.jpg')
#Image.open('PCA Logistic Regression.png').save('PCA Logistic Regression.jpg','JPEG')
#
plt.figure(3)
y2=np.array(hold2)
plt.plot(x,y2)
plt.xticks(np.arange(x.min(), x.max(), 1))
plt.xlabel('Number of Component')
plt.ylabel('Error%')
plt.title('Linear Regression Training Set')
plt.savefig('PCA LinearRegression.jpg')
#
plt.figure(4)
y3=np.array(hold3)
plt.plot(x,y3)
plt.xlabel('Number of Component')
plt.xticks(np.arange(x.min(), x.max(), 1))
plt.ylabel('Error%')
plt.title('Polynamial Regression Degree 6 Training Set')
plt.savefig('PCA Poly. Regr. Degree 6.jpg')
#
from itertools import chain
#index_min = np.argmin(hold,hold2,hold3)
mh=min(chain(hold,hold2,hold3))
mh1=min(chain(hold,hold2))
#minumum each
ff1=min(chain(hold))
ind1=hold.index(ff1)+1
ff2=min(hold2)
ind2=hold2.index(ff2)+1
ff3=min(hold3)
ind3=hold3.index(ff3)+1
"""
if mh in hold:
    f=hold.index(mh)+1
    jj=1    
elif mh in hold2:
    f=hold2.index(mh)+1
    jj=2
elif mh in hold3:
    f=hold3.index(mh)+1
    jj=3
    
if mh1 in hold:
    f2=hold.index(mh1)+1
    jj3=1    
elif mh in hold2:
    f2=hold2.index(mh1)+1
    jj3=2
"""
    
    
holdT=[]
holdT2=[]   
holdT3=[]
dataset = pd.read_csv('german_credit.csv')
XV= dataset.iloc[:, 1:].values
yV = dataset.iloc[:, 0].values
pca = PCA(n_components = ind1)
pca1 = PCA(n_components = ind2)
pca2 = PCA(n_components = ind3)
for k in range(100):
    X_trainF, X_testF, y_trainF, y_testF = train_test_split(XV, yV, test_size = 0.3, random_state = k)
    X_trainF = sc.fit_transform(X_trainF)
    X_testF = sc.transform(X_testF) 
    # logistic regresion
    X_trainFo=pca.fit_transform(X_trainF)
    X_testFo=pca.transform(X_testF)    
    classifier.fit(X_trainFo, y_trainF)
    y_predT = classifier.predict(X_testFo)
    errorT = np.mean( y_testF != y_predT)
    holdT.append(errorT)
    # linear Regression
    
    X_trainFl=pca1.fit_transform(X_trainF)
    X_testFl=pca1.transform(X_testF)     
    regressor.fit(X_trainFl, y_trainF)    
    y_predTl2= regressor.predict(X_testFl)
    y_predTl2 = (y_predTl2 > 0.5)
    errorT2 = np.mean( y_testF != y_predTl2)
    holdT2.append(errorT2)
   # polynamial regression  
    
    X_trainFp=pca2.fit_transform(X_trainF)
    X_testFp=pca2.transform(X_testF)   
    poly_reg = PolynomialFeatures(degree = 6)
    X_polyp = poly_reg.fit_transform(X_trainFp)
    lin_reg_2.fit(X_polyp, y_trainF)
    y_predT3p= lin_reg_2.predict(poly_reg.fit_transform(X_testFp))#lin_reg_2.predict(X_test)
    y_predT3p = (y_predT3p > 0.5)
    errorT3 = np.mean(y_testF != y_predT3p)
    holdT3.append(errorT3)
    del X_trainF
    del X_testF
    del y_trainF 
    del y_testF
    del y_predT
    del y_predTl2
    del y_predT3p


xd= np.arange(1,101)
y8=np.array(holdT)
meaz=np.mean(y8)
cvar=np.var(y8)
np.save('testmevarlog',meaz,cvar)
## explained variance
plt.figure(8)
plt.plot(xd,y8)
#plt.xticks(np.arange(xd.min(), xd.max(), 1))
#plt.yticks(np.arange(y.min()-0.005, y.max()+0.05, 0.05))
plt.xlabel('Number of Trial')
plt.ylabel('Error')
plt.title('Log. Reg. Test Set PCA %i' %ind1)
plt.savefig('PCA Logistic Regression Test.jpg')

#Image.open('PCA Variance.png').save('PCA Variance.jpg','JPEG')
y9=np.array(holdT2)
meaz2=np.mean(y9)
cvar2=np.var(y9)
np.save('testmevarline',meaz2,cvar)
plt.figure(9)
plt.plot(xd,y9)

plt.xlabel('Number of Trial')
plt.ylabel('Error')
plt.title('Lin. Reg. Test Set PCA %i' %ind2)
plt.savefig('PCA LinearRegression test.jpg')
# polynamial regression 
y10=np.array(holdT3)
meaz3=np.mean(y10)
cvar3=np.var(y10)
np.save('polyre',meaz3,cvar3)
plt.figure(10)
plt.plot(xd,y10)
plt.xlabel('Number of Trial')
plt.ylabel('Error')
plt.title('Poly. Reg. Test Set PCA %i' %ind3)
plt.savefig('PCA Poly regres test.jpg')
