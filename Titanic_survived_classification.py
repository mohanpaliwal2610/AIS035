#Name: Mohan Subhash Paliwal..
#ID: AIS035
#------------------------------------------------------------------------------
# data set 1 :-Titanic : Machine learning from Disaster
"""
Goal

It is your job to predict if a passenger survived the sinking of the Titanic or not.
For each in the test set, you must predict a 0 or 1 value for the variable.

here we used many classification algorithm and compare the accuracy of model and select ramdom forest model for prediction.
"""
# -----rread data------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------
data=pd.read_csv("C:\\Users\\admin\\Desktop\\titanic\\train.csv")
print(data.head())
'''
 PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S

[5 rows x 12 columns]
'''
#------------------------------------------------------

#Check the head of customers, and check out its info() and describe() methods.

print(data.info())
print(data.describe())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 66.2+ KB
None
       PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200


'''

#==================DATA PREPROCESSING AND CLEANING================================
data.drop("PassengerId",axis=1,inplace=True)
data.drop("Name",axis=1,inplace=True)
data.drop("Ticket",axis=1,inplace=True)
data.drop("Cabin",axis=1,inplace=True)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(data["Sex"])
data["Sex"]=le.transform(data["Sex"])
print data.head()

data["Age"].hist(bins=20)
plt.show()

print(data["Age"].isnull().sum())
data["Age"].fillna(data["Age"].mean(),inplace=True)
print(data["Age"].isnull().sum())

data["Fare"].hist(bins=20)
plt.show()

print(data["Fare"].isnull().sum())
data["Fare"].fillna(data["Fare"].mode(),inplace=True)
print(data["Fare"].isnull().sum())

print(data["Embarked"].value_counts())
data["Embarked"].fillna("S",inplace=True)
print(data["Embarked"].isnull().sum())
print(data["Embarked"].value_counts())

le.fit(data["Embarked"])
data["Embarked"]=le.transform(data["Embarked"])

print(data.head())
print(data.columns)

#------Model Building------------------

#---------seperate data--------------------------
x=data.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y=data.loc[:,['Survived']].values.reshape(-1,1)

#------traon test split----------------------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=101)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)


#================================================
#model fitting------------------
'''
#----------model import---KNeighborsClassifier---------------------------
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=15)                    # accuracy 69.75%
#--------fit model------------------------------------
knn.fit(x_train,y_train)

#-------predict data test-----------------------------
y_pred=knn.predict(x_test)
print(y_pred)
#------creat confusion matrix-------------------------
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#------check accuracy --------------------------------
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)
'''
'''

#==========
#============Naive Bayes Classification ==using GaussianNB
from sklearn.naive_bayes import GaussianNB                 # accuracy 76.86567164179104
gm=GaussianNB()
gm.fit(x_train,y_train)
y_pred=gm.predict(x_test)
#------creat confusion matrix-------------------------
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#------check accuracy -----------------------------
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)
'''


'''
#===============Decision Tree Classification==============
from sklearn.tree import DecisionTreeClassifier
gm=DecisionTreeClassifier()                                  #accuracy 79.10447761194029 using ginni index
#gm=DecisionTreeClassifier(criterion="entropy",max_depth=1000)   # accuracy 79.47761194029851 using entropy
gm.fit(x_train,y_train)
y_pred=gm.predict(x_test)
#------creat confusion matrix-------------------------
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#------check accuracy -----------------------------
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)
'''

#----------model import---Random Forest Classification---------------------------
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100)                    # accuracy 80.59701492537313
#--------fit model------------------------------------
rf.fit(x_train,y_train)

#-------predict data test-----------------------------
y_pred=rf.predict(x_test)
print(y_pred)
#------creat confusion matrix-------------------------
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
#------check accuracy --------------------------------
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)
#================OUTPUT===============
'''
[0 1 0 1 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 1 0 0 1 1 0 0 0 0 0 0 1 0 1 1 0 1 0
 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 0
 0 1 0 0 0 0 0 1 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 0 0 1 0 1 0 0 1 1 1 1 0 0 0
 1 1 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 1 0 1 0 0 1 0 1 0
 0 0 1 0 1 0 0 1 1 0 1 1 0 0 0 0 0 1 0 1 0 0 1 0 1 1 0 0 1 0 0 0 1 0 1 1 1
 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 1 1 1
 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 1 1 0 1 1 0 0 1 1 0 0 1 0 0 1 0 0 0 0 0 0 1
 0 0 1 1 0 1 1 0 1]
[[136  18]
 [ 32  82]]
0.8134328358208955
              precision    recall  f1-score   support

           0       0.81      0.88      0.84       154
           1       0.82      0.72      0.77       114

   micro avg       0.81      0.81      0.81       268
   macro avg       0.81      0.80      0.81       268
weighted avg       0.81      0.81      0.81       268
'''


var=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
#======Identify Importance of Variables=========================
imp=pd.DataFrame(rf.feature_importances_,index=x_train.columns,columns=["importance"]).sort_values("importance",ascending=False)
print(imp)
from sklearn.feature_selection import SelectFromModel
sdt=SelectFromModel(rf,threshold=0.15)

sdt.fit(x_train,y_train)

#print name name of importtant variables
for i in sdt.get_support(indices=True):
    print(var[i])
#-------predict data test-----------------------------
y_pred=rf.predict(x_test)
print(y_pred)
#------creat confusion matrix-------------------------
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#------check accuracy --------------------------------
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)

'''====IMPORTANCE OF VARIABLES===========
          importance
Sex         0.263850
Fare        0.263654
Age         0.263492
Pclass      0.082306
SibSp       0.049699
Parch       0.039939
Embarked    0.037061
[[135  19]
 [ 37  77]]
0.7910447761194029


'''


'''
#=====Hard Voting====================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
knn_2=KNeighborsClassifier(n_neighbors=2)
knn_3=KNeighborsClassifier(n_neighbors=3)
knn_5=KNeighborsClassifier(n_neighbors=10)
knn_7=KNeighborsClassifier(n_neighbors=15)
rf=RandomForestClassifier(n_estimators=120,random_state=101)
dt=DecisionTreeClassifier()
gnb=GaussianNB()

voting_Hard=VotingClassifier(estimators=[("GNB",gnb),("RF",rf),("DT",dt),("KNN_5",knn_5),("KNN_7",knn_7)],voting="hard")

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
'''



#================================PREDICTION FOR TEST DATA===========================================
#for test dataprediction
data=pd.read_csv("C:\\Users\\admin\\Desktop\\titanic\\test.csv")

#------------------------------------------------------

#Check the head of customers, and check out its info() and describe() methods.
print(data.info())
print(data.describe())

#==================================================
PassengerId=data["PassengerId"]
data.drop("PassengerId",axis=1,inplace=True)
data.drop("Name",axis=1,inplace=True)
data.drop("Ticket",axis=1,inplace=True)
data.drop("Cabin",axis=1,inplace=True)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(data["Sex"])
data["Sex"]=le.transform(data["Sex"])
print data.head()

data["Age"].hist(bins=20)
plt.show()

print(data["Age"].isnull().sum())
data["Age"].fillna(data["Age"].mean(),inplace=True)
print(data["Age"].isnull().sum())

data["Fare"].hist(bins=20)
plt.show()
print(data["Fare"].mode())
print(data["Fare"].isnull().sum())
data["Fare"].fillna(data["Fare"].mode(),inplace=True)
print(data["Fare"].isnull().sum())


print(data["Embarked"].value_counts())
data["Embarked"].fillna("S",inplace=True)
print(data["Embarked"].isnull().sum())
print(data["Embarked"].value_counts())

le.fit(data["Embarked"])
data["Embarked"]=le.transform(data["Embarked"])

print(data.head())
print(data.columns)


x=data.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y_pred=rf.predict(x)


print(y_pred)
print(len(y_pred))
print(type(y_pred))

submission=pd.DataFrame({"PassengerId":PassengerId,"Survived":y_pred})

submission.to_csv("C:\\Users\\admin\\Desktop\\titanic\\sample_submission1.csv",index=False)

