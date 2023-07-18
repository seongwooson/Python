from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.6, random_state=42)
svm = svm.SVC(kernel='linear', C=1.0, gamma=.5)
svm.fit(X_train, y_train)
prediction = svm.predict(X_test)
score = metrics.accuracy_score(y_test, prediction)
print('정확도: {0:f}'.format(score))
