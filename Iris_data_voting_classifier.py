#Name: Mohan Subhash Paliwal..
#ID: AIS035
#------------------------------------------------------------------------------
# data set 1 :- iris
# -----rread data------------------------
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
import seaborn as sns
#iris=sns.load_dataset("iris")
#iris.head()

iris=pd.read_csv("C:\\Users\\admin\\Desktop\\AIS Solutions\\iris.csv")
print(iris.head())
#---------seperate data--------------------------
x=iris.iloc[:,0:4]
y=iris.iloc[:,4:5]
print y.head()
#------traon test split----------------------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)

#----------model import------------------------------
#=====Hard Voting====================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
knn_2=KNeighborsClassifier(n_neighbors=2)
knn_3=KNeighborsClassifier(n_neighbors=3)
knn_5=KNeighborsClassifier(n_neighbors=5)
knn_7=KNeighborsClassifier(n_neighbors=7)
rf=RandomForestClassifier()
dt=DecisionTreeClassifier()
gnb=GaussianNB()

voting_Hard=VotingClassifier(estimators=[("GNB",gnb),("RF",rf),("DT",dt),("KNN_2",knn_2),("KNN_3",knn_3),("KNN_5",knn_5),("KNN_7",knn_7)],voting="hard")


#--------fit model----------------------------------
voting_Hard.fit(x_train,y_train)

#-------predict data test-----------------------------
y_pred=voting_Hard.predict(x_test)
print(y_pred)
#------creat confusion matrix-------------------------
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#------check accuracy -----------------------------
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)
#====OUTPUT FOR HARD VOTING CLASSIFIER=========
'''
  Sepal.Length  Sepal.Width  Petal.Length  Petal.Width Species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
('x_train :', (120, 4))
('y_train :', (120, 1))
('x_test :', (30, 4))
('y_test :', (30, 1))
C:\Python27\lib\site-packages\sklearn\preprocessing\label.py:219: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
C:\Python27\lib\site-packages\sklearn\preprocessing\label.py:252: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
C:\Python27\lib\site-packages\sklearn\ensemble\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
['setosa' 'setosa' 'setosa' 'virginica' 'versicolor' 'virginica'
 'versicolor' 'versicolor' 'virginica' 'setosa' 'virginica' 'setosa'
 'setosa' 'virginica' 'virginica' 'versicolor' 'versicolor' 'versicolor'
 'setosa' 'virginica' 'versicolor' 'setosa' 'versicolor' 'versicolor'
 'versicolor' 'versicolor' 'versicolor' 'virginica' 'setosa' 'setosa']
[[10  0  0]
 [ 0 12  0]
 [ 0  0  8]]
1.0
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        12
   virginica       1.00      1.00      1.00         8

   micro avg       1.00      1.00      1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


Process finished with exit code 0

'''
#=================SOFT VOTING CLASSIFIER====================
voting_soft=VotingClassifier(estimators=[("GNB",gnb),("RF",rf),("DT",dt),("KNN_2",knn_2),("KNN_3",knn_3),("KNN_5",knn_5),("KNN_7",knn_7)],voting="soft")


#--------fit model----------------------------------
voting_soft.fit(x_train,y_train)

#-------predict data test-----------------------------
y_pred=voting_soft.predict(x_test)
print(y_pred)
#------creat confusion matrix-------------------------
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#------check accuracy -----------------------------
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)

#====OUTPUT FOR SOFT VOTING===============
'''
['setosa' 'setosa' 'setosa' 'virginica' 'versicolor' 'virginica'
 'versicolor' 'versicolor' 'virginica' 'setosa' 'virginica' 'setosa'
 'setosa' 'virginica' 'virginica' 'versicolor' 'versicolor' 'versicolor'
 'setosa' 'virginica' 'versicolor' 'setosa' 'versicolor' 'versicolor'
 'versicolor' 'versicolor' 'versicolor' 'virginica' 'setosa' 'setosa']
[[10  0  0]
 [ 0 12  0]
 [ 0  0  8]]
1.0
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        12
   virginica       1.00      1.00      1.00         8

   micro avg       1.00      1.00      1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


Process finished with exit code 0

'''