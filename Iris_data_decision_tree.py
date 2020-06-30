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
#=====DecisionTreeClassifier======using entropy=======
from sklearn.tree import DecisionTreeClassifier
#dt=DecisionTreeClassifier("entropy")                            #Accuracy 0.9666666666666667

dt=DecisionTreeClassifier()                                      #Accuracy 0.9666666666666667 using ginni index
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

#---------------------------------------------------------------------------

#================================OUTPUT=========================================
'''
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
['setosa' 'setosa' 'setosa' 'versicolor' 'versicolor' 'virginica'
 'versicolor' 'versicolor' 'virginica' 'setosa' 'virginica' 'setosa'
 'setosa' 'virginica' 'virginica' 'versicolor' 'versicolor' 'versicolor'
 'setosa' 'virginica' 'versicolor' 'setosa' 'versicolor' 'versicolor'
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


Process finished with exit code 0

'''