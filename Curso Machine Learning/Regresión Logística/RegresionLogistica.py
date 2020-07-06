#Regresión Logística

########################################Libraries##########################################
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

#Because the data has different scale, we have to put in the same one
escalar = StandardScaler()
x_train = escalar.fit_transform(x_train)
x_test = escalar.transform(x_test)

#Define algorithm
algoritmo = LogisticRegression()
algoritmo.fit(x_train, y_train)

#Make a prediction
y_pred = algoritmo.predict(x_test)

#Verifing matrix
matriz = confusion_matrix(y_test, y_pred)
print("Matrix:")
print(matriz)


#Model precision
precision = precision_score(y_test, y_pred)
print("Model Precision")
print(precision)


#Accuracy
exactitud = accuracy_score(y_test, y_pred)
print("Accurracy: ", exactitud)

#Recall
sens = recall_score(y_test, y_pred)
print("Recall: ", sens)


#F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score: ", f1)

#ROC
ROC = roc_auc_score(y_test, y_pred)
print("ROC: ", ROC)
######################################Prepare Data#########################################