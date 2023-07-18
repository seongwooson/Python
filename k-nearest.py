import numpy as np #vector, matrix calculation
import matplotlib.pyplot as plt #draw data into chart, plot
import pandas as pd #data analyze and control
from sklearn import metrics #model evalutate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv('C:/Users/tjddn/Desktop/pytorch/chap03/data/iris.data')

x = dataset.iloc[:,:-1].values #use all column, except last data of row
y = dataset.iloc[:, 4].values #use all column, and use fifth datat of row
#print(x)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

s = StandardScaler() #scaler mean = 0, deviation = 1

s.fit(X_train)

X_train = s.transform(X_train)
X_test = s.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("accuracy: {}".format(accuracy_score(y_test, y_pred)))

k = 10
acc_array = np.zeros(k)
for k in np.arange(1, k+1, 1):
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k-1] = acc

max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print("accuracy: ", max_acc, "k = ", k+1)

