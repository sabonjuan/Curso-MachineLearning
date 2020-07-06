#Bosque Aleatorio de Regresión

########################################Libraries##########################################
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
######################################Prepare Data#########################################

#Import data from sikvit-learn library
boston = datasets.load_boston()
# print(boston)
# print()

# #Get some features from dataset.
# print("Keys from the datasets")
# print(boston.keys())
# print()

# print("Data set's description")
# print(boston.DESCR)
# print()

# print("Amounth of data")
# print(boston.data.shape)
# print()

# print("feature's name")
# print(boston.feature_names)
# print()

#############################Prepare data for the algorithm################################
#Select column 6 from dataset
x_bar = boston.data[:, np.newaxis, 5]

#Define the data of the corresponding labels
y_bar = boston.target

# #Plot data
# plt.scatter(x_bar, y_bar)
# plt.show()

###################################Implement model#########################################
x_train, x_test, y_train, y_test = train_test_split(x_bar, y_bar, test_size=0.2)

#Define algorithm to use
bar = RandomForestRegressor(n_estimators=150, max_depth=25)

#Train model
bar.fit(x_train, y_train)

#Make predictions
y_pred = bar.predict(x_test)

#Plot test data with predictions
# x_grid = np.arange(min(x_test), max(x_test), 0.1)
# x_grid = x_grid.reshape((len(x_grid), 1))
# plt.scatter(x_test, y_test)
# plt.plot(x_grid, bar.predict(x_grid),color='red',linewidth=3)
# plt.show()

#Data model's review
print("Data's model review")
print("Accuracy: ", bar.score(x_train, y_train))