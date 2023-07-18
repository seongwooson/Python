import pandas as pd
df = pd.read_csv('C:/Users/tjddn/Desktop/pytorch/chap03/data/titanic/train.csv', index_col = 'PassengerId')
print(df.head())

df = df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df['Sex'] = df['Sex'].map({'male': 0, 'femail': 1})
df = df.dropna() #delete NA data
X = df.drop('Survived', axis=1) 
y = df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn import tree
model = tree.DecisionTreeClassifier() #criterion 'gini', 'entropy', 'log_loss' default:'gini'

model.fit(X_train, y_train)

y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))