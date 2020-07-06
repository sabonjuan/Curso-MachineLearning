#Arbol de decisión (Regresión)

#Regresión Lineal Múltiple


########################################Libraries##########################################
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

#############################Prepare data for the algorithm################################
#Select column 6 from dataset
x_adr = boston.data[:, np.newaxis, 5]

#Define the data of the corresponding labels
y_adr = boston.target

#Plot data
plt.scatter(x_adr, y_adr)
plt.show()

###################################Implement model#########################################
x_train, x_test, y_train, y_test = train_test_split(x_adr, y_adr, test_size=0.2)

#Define algorithm to use
adr = DecisionTreeRegressor(max_depth = 15)

#Train the model
adr.fit(x_train, y_train)

#Make a prediction
y_pred = adr.predict(x_test)

#Plot training data
x_grid = np.arange(min(x_test), max(x_test), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x_test, y_test)
plt.plot(x_grid, adr.predict(x_grid), color='red', linewidth=3)
plt.show()

print("Data's model review")
print("Accuracy: ", adr.score(x_train, y_train))