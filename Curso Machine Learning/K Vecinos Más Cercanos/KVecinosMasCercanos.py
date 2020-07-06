#K Vecinos Más Cercanos

########################################Libraries##########################################
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score

######################################Prepare Data#########################################
dataset = datasets.load_breast_cancer()

#print(dataset)

##################################Undetstading Dataset#####################################
# #Verify dataset information
# print('Dataset information:')
# print(dataset.keys())
# print()

# #Verify dataset characteristic
# print(dataset.DESCR)
# print()

##Because we have all data, we don't need fit them

#Select all columns
x = dataset.data

#Define the data corresponding to labels
y = dataset.target

#####################################Implementation########################################
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

algoritmo = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

algoritmo.fit(x_train, y_train)

y_pred = algoritmo.predict(x_test)

#Verifing Matriz
matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(matrix)


#Presicion
precision = precision_score(y_test, y_pred)
print("Presicion: ", precision)

##################################Undetstading Dataset#####################################