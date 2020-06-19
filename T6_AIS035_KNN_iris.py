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

iris=pd.read_csv("C:\\Users\\admin\\Desktop\\iris.csv")
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
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
#--------fit model----------------------------------
knn.fit(x_train,y_train)

#-------predict data test-----------------------------
y_pred=knn.predict(x_test)
print(y_pred)
#------creat confusion matrix-------------------------
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#------check accuracy -----------------------------
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)

#---------------------------------------------------------------------------