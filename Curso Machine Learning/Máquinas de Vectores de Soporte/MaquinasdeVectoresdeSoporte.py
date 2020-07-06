#Máquinas de Vectores de Soporte

########################################Libraries##########################################
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
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
#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Define algorithm
algoritmo = SVC(kernel = 'linear')

#Train
algoritmo.fit(x_train, y_train)

#Make prediction
y_pred = algoritmo.predict(x_test)

#Verifing matrix
matriz  = confusion_matrix(y_test, y_pred)
print("matrix")
print(matriz)

#Precision
precision = precision_score(y_test, y_pred)
print("Precision: ", precision)

