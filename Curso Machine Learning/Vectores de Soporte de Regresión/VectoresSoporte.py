#Vectores de Soporte Regresión

########################################Libraries##########################################
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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


################################Prepare List of Vectors####################################

#Select column 6 of Boston Houses Dataset
x_svr = boston.data[:, np.newaxis, 5]

#define data
y_svr = boston.target

#Plot data
plt.scatter(x_svr, y_svr)
plt.show()

##################################Implementation of SVR####################################
""" 
As we can see in the previos plott, all data must be linear, so we will set the kernel with
this parametes
"""

#Split dan on train and test
x_train, x_test, y_train, y_test = train_test_split(x_svr, y_svr, test_size=0.2)

#Define algorithm
#svr = SVR(kernel='linear', C=1.0, epsilon=0.2)
svr = SVR()
#Train model
svr.fit(x_train, y_train)

#Make prediction
y_pred = svr.predict(x_test)

#Plot data with model
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='red', linewidth=3)
plt.show()

print()
print('Description of SVR Linear Model')
print('Accuracy= ', svr.score(x_train, y_train))

