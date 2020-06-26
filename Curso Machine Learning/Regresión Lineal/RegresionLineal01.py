#Regresión Lineal Simple


########################################Libraries##########################################
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

######################################Prepare Data#########################################

#Import data from sikvit-learn library
boston = datasets.load_boston()
print(boston)
print()

#Get some features from dataset.
print("Keys from the datasets")
print(boston.keys())
print()

print("Data set's description")
print(boston.DESCR)
print()

print("Amounth of data")
print(boston.data.shape)
print()

print("feature's name")
print(boston.feature_names)
print()

#Get data from dataset
x = boston.data[:, np.newaxis,5]
y = boston.target

#Plott Data
plt.scatter(x,y)
plt.xlabel("Room numbers")
plt.ylabel("Avg. Values")
plt.show()

##########################Linear Regretion's implementatation##############################
#Split data from "train" in traingin and testing for test the algorithm

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#Define algorithm to use
lr = linear_model.LinearRegression()

#Traine the model --> Ver traducción correcta
lr.fit(x_train, y_train)

#Predict
y_pred = lr.predict(x_test)

#Plott results from traine vs predict
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='red', linewidth=3)
plt.title('Linear Regression')
plt.xlabel('Room Numbers')
plt.ylabel('Avg. Value')
plt.show()

#Linear regression model's description
print('Model Description y=ax+b')
print('coef a: ', lr.coef_)
print('coef b: ', lr.intercept_)
print()
print('The models equation is: ')
print('y= ', lr.coef_,'x + ',lr.intercept_)

#Model presicion
print()
print('Model Presicion:')
print(lr.score(x_train, y_train))