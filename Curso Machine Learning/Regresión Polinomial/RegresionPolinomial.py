#Regresión polinomial

########################################Libraries##########################################
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
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

#################################Polynomial Regression ####################################

#Select 6th column, remember that column numbers start from 0
x_p = boston.data[:, np.newaxis, 5]
y_p = boston.target

#Plot data
plt.scatter(x_p, y_p)
plt.show()

#Split data
x_train, x_test, y_train, y_test = train_test_split(x_p, y_p, test_size=0.2)

#Define polynomial degree
poli_reg = PolynomialFeatures(degree = 2)

#Now, proceed to transform current characteristics.
x_train_poli = poli_reg.fit_transform(x_train)
x_test_poli = poli_reg.fit_transform(x_test)

#define algorithm
pr = linear_model.LinearRegression()

#traine model
pr.fit(x_train_poli, y_train)

#Make prediction
y_pred_pr = pr.predict(x_test_poli)

#Graficamos los datos
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred_pr, color='red', linewidth=3)
plt.show()

#Coefficients
print()
print("Model's description")
print()
print("value of 'a ': ", pr.coef_)
print("Value of 'b': ", pr.intercept_)
print('Algorithtm precission: ', pr.score(x_train_poli, y_train) )

