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
var=["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]
#------traon test split----------------------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)

#----------model import------------------------------
#=====Random_forest_Classifier======u=======
from sklearn.ensemble import RandomForestClassifier

dt=RandomForestClassifier(n_estimators=100,random_state=101)
#--------fit model----------------------------------
dt.fit(x_train,y_train)

#-------predict data test-----------------------------
y_pred=dt.predict(x_test)
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

#OUTPUT==============
"""
   Sepal.Length  Sepal.Width  Petal.Length  Petal.Width Species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
  Species
0  setosa
1  setosa
2  setosa
3  setosa
4  setosa
('x_train :', (120, 4))
('y_train :', (120, 1))
('x_test :', (30, 4))
('y_test :', (30, 1))
C:/Users/admin/PycharmProjects/AIS_solutions/Iris_data_random_forest.py:36: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  dt.fit(x_train,y_train)
['setosa' 'setosa' 'setosa' 'virginica' 'versicolor' 'virginica'
 'versicolor' 'versicolor' 'virginica' 'setosa' 'virginica' 'setosa'
 'setosa' 'virginica' 'virginica' 'versicolor' 'versicolor' 'versicolor'
 'setosa' 'versicolor' 'versicolor' 'setosa' 'versicolor' 'versicolor'
 'versicolor' 'versicolor' 'versicolor' 'virginica' 'setosa' 'setosa']
[[10  0  0]
 [ 0 12  0]
 [ 0  1  7]]
0.9666666666666667
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.92      1.00      0.96        12
   virginica       1.00      0.88      0.93         8

   micro avg       0.97      0.97      0.97        30
   macro avg       0.97      0.96      0.96        30
weighted avg       0.97      0.97      0.97        30


"""

#======Identify Importance of Variables=========================
imp=pd.DataFrame(dt.feature_importances_,index=x_train.columns,columns=["importance"]).sort_values("importance",ascending=False)
print(imp)
"""
              importance
Petal.Width     0.434920
Petal.Length    0.418405
Sepal.Length    0.118229
Sepal.Width     0.028446


"""
from sklearn.feature_selection import SelectFromModel
sdt=SelectFromModel(dt,threshold=0.15)

sdt.fit(x_train,y_train)

#print name name of importtant variables
for i in sdt.get_support(indices=True):
    print(var[i])
    '''
    Petal.Length
    Petal.Width
'''
#creat a data subset with only most imp variables
x_train=sdt.transform(x_train)
x_test=sdt.transform(x_test)
print(x_train.shape)

#Train the new Random Forest Classifier Using only important Variables
#----------model import------------------------------
#=====Random_forest_Classifier======u=======
from sklearn.ensemble import RandomForestClassifier

dt=RandomForestClassifier(n_estimators=100,random_state=101)                                      #Accuracy 0.9666666666666667 using ginni index
#--------fit model----------------------------------
dt.fit(x_train,y_train)

#-------predict data test-----------------------------
y_pred=dt.predict(x_test)
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
#OUTPUT=======
'''
(120, 2)
C:/Users/admin/PycharmProjects/AIS_solutions/Iris_data_random_forest.py:88: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  dt.fit(x_train,y_train)
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