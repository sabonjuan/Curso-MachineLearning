#Regresión Lineal Múltiple


########################################Libraries##########################################
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

######################################Prepare Data#########################################
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

##########################Linear Regretion's implementatation##############################

x_multiple = boston.data[:,5:8]
print(x_multiple)

#Define data from target
y_multiple = boston.target

#Split data from "train" in traingin and testing for test the algorithm
x_train, x_test, y_train, y_test = train_test_split(x_multiple,y_multiple,test_size=0.2)
lr_multiple = linear_model.LinearRegression()

#train the model
lr_multiple.fit(x_train, y_train)

#Make predictions
y_pred_multiple = lr_multiple.predict(x_test)

#Data model
print('Data model is y = ax1 + ax2 + ax3 + b')
print()
print('All a values: ', lr_multiple.coef_)

print('b = ',lr_multiple.intercept_)

#Model Precission
print("Model pressicion")
print(lr_multiple.score(x_train, y_train))
