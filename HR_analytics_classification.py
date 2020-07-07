#Name: Mohan Subhash Paliwal..
#ID: AIS035
#------------------------------------------------------------------------------
# data set 1 :- HR Analytics Data

"""
About Practice Problem: HR Analytics

HR analytics is revolutionising the way human resources departments operate, leading to higher efficiency
and better results overall. Human resources has been using analytics for years. However, the collection, processing
and analysis of data has been largely manual, and given the nature of human resources dynamics and HR KPIs,
the approach has been constraining HR. Therefore, it is surprising that HR departments woke up to the utility
of machine learning so late in the game.
Here is an opportunity to try predictive analytics in identifying the employees most likely to get promoted.
"""

# -----rread data------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------
data=pd.read_csv("C:\\Users\\admin\\Desktop\\HR_Analytics\\train_LZdllcl.csv")
print(data.head())

#------------------------------------------------------

#Check the head of data and check out its info() and describe() methods.

print(data.info())
print(data.describe())
"""===OUTPUT====
  employee_id         department  ... avg_training_score is_promoted
0        65438  Sales & Marketing  ...                 49           0
1        65141         Operations  ...                 60           0
2         7513  Sales & Marketing  ...                 50           0
3         2542  Sales & Marketing  ...                 50           0
4        48945         Technology  ...                 73           0

[5 rows x 14 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 54808 entries, 0 to 54807
Data columns (total 14 columns):
employee_id             54808 non-null int64
department              54808 non-null object
region                  54808 non-null object
education               52399 non-null object
gender                  54808 non-null object
recruitment_channel     54808 non-null object
no_of_trainings         54808 non-null int64
age                     54808 non-null int64
previous_year_rating    50684 non-null float64
length_of_service       54808 non-null int64
KPIs_met >80%           54808 non-null int64
awards_won?             54808 non-null int64
avg_training_score      54808 non-null int64
is_promoted             54808 non-null int64
dtypes: float64(1), int64(8), object(5)
memory usage: 4.8+ MB
None
        employee_id  no_of_trainings  ...  avg_training_score   is_promoted
count  54808.000000     54808.000000  ...        54808.000000  54808.000000
mean   39195.830627         1.253011  ...           63.386750      0.085170
std    22586.581449         0.609264  ...           13.371559      0.279137
min        1.000000         1.000000  ...           39.000000      0.000000
25%    19669.750000         1.000000  ...           51.000000      0.000000
50%    39225.500000         1.000000  ...           60.000000      0.000000
75%    58730.500000         1.000000  ...           76.000000      0.000000
max    78298.000000        10.000000  ...           99.000000      1.000000

"""
#===============DATA PREPROCESSING AND CLEANING===================================
data.drop("employee_id",axis=1,inplace=True)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(data["region"])
data["region"]=le.transform(data["region"])
print data.head()

le.fit(data["gender"])
data["gender"]=le.transform(data["gender"])
print(len(data["gender"]))

le.fit(data["recruitment_channel"])
data["recruitment_channel"]=le.transform(data["recruitment_channel"])

le.fit(data["department"])
data["department"]=le.transform(data["department"])

print(data["education"].value_counts())
print(data["education"].isnull().sum())
data["education"].fillna("Bachelor's",inplace=True)
print(data["education"].value_counts())
print(data["education"].isnull().sum())

le.fit(data["education"])
data["education"]=le.transform(data["education"])

print(len(data["gender"]))


data["previous_year_rating"].hist(bins=20)
plt.show()

print(data["previous_year_rating"].value_counts())
data["previous_year_rating"].fillna(data["previous_year_rating"].mean(),inplace=True)
print(data["previous_year_rating"].isnull().sum())

print(data.columns)
x=data.loc[:,['department','region','education','gender','recruitment_channel','no_of_trainings','age',
       'previous_year_rating','length_of_service','KPIs_met >80%',
       'awards_won?','avg_training_score']]
y=data.loc[:,['is_promoted']].values.reshape(-1,1)

#------train test split----------------------------

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)
'''==OUTPUT===
('x_train :', (43846, 12))
('y_train :', (43846, 1))
('x_test :', (10962, 12))
('y_test :', (10962, 1))
'''
"""
#==========Over ampling========================
from imblearn.over_sampling import SMOTE
smt=SMOTE()
x_train,y_train=smt.fit_sample(x_train,y_train)

print("after Over sampling count of label :1 is :",sum(y_train==1))
print("after Over sampling count of label :0 is :",sum(y_train==0))
"""
"""
#====Under sampling============================
from imblearn.under_sampling import NearMiss
nm=NearMiss()
x_train,y_train=nm.fit_sample(x_train,y_train)

print("after Over sampling count of label :1 is :",sum(y_train==1))
print("after Over sampling count of label :0 is :",sum(y_train==0))

"""

#----------model import------------------------------

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
'''===OUTPUT=========
[0 0 0 ... 0 0 0]
[[9989   51]
 [ 676  246]]
0.9336799854041233    # ACCURACY
              precision    recall  f1-score   support

           0       0.94      0.99      0.96     10040
           1       0.83      0.27      0.40       922

   micro avg       0.93      0.93      0.93     10962
   macro avg       0.88      0.63      0.68     10962
weighted avg       0.93      0.93      0.92     10962

'''
imp=pd.DataFrame(dt.feature_importances_,index=x_train.columns,columns=["importance"]).sort_values("importance",ascending=False)
print(imp)
'''
                      importance
avg_training_score      0.293913
age                     0.144529
region                  0.123810
length_of_service       0.107936
department              0.096149
previous_year_rating    0.058693
KPIs_met >80%           0.046790
awards_won?             0.032164
recruitment_channel     0.031249
no_of_trainings         0.024872
gender                  0.024393
education               0.015501

'''


#-------------------------------------------------------------------------------------------
#for test dataprediction
data=pd.read_csv("C:\\Users\\admin\\Desktop\\HR_Analytics\\test_2umaH9m.csv")

#------------------------------------------------------

employee_id=data["employee_id"]
#Check the head of customers, and check out its info() and describe() methods.
#==================================================
data.drop("employee_id",axis=1,inplace=True)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(data["region"])
data["region"]=le.transform(data["region"])
print data.head()

le.fit(data["gender"])
data["gender"]=le.transform(data["gender"])
print(len(data["gender"]))

le.fit(data["recruitment_channel"])
data["recruitment_channel"]=le.transform(data["recruitment_channel"])

le.fit(data["department"])
data["department"]=le.transform(data["department"])


print(data["education"].value_counts())
print(data["education"].isnull().sum())
data["education"].fillna("Bachelor's",inplace=True)
print(data["education"].value_counts())
print(data["education"].isnull().sum())

le.fit(data["education"])
data["education"]=le.transform(data["education"])

print(len(data["gender"]))


data["previous_year_rating"].hist(bins=20)
plt.show()

print(data["previous_year_rating"].value_counts())
data["previous_year_rating"].fillna(data["previous_year_rating"].mean(),inplace=True)
print(data["previous_year_rating"].isnull().sum())

print(data.columns)
x=data.loc[:,['department','region','education','gender','recruitment_channel','no_of_trainings','age',
       'previous_year_rating','length_of_service','KPIs_met >80%',
       'awards_won?','avg_training_score']]

#==================================================
y_pred=dt.predict(x)
'''
'''
print(y_pred)
print(len(y_pred))
print(type(y_pred))

submission=pd.DataFrame({"employee_id":employee_id,"is_promoted":y_pred})

submission.to_csv("C:\\Users\\admin\\Desktop\\HR_Analytics\\sample_submission1.csv",index=False)







'''


#=====DecisionTreeClassifier======using entropy=======
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
rf=RandomForestClassifier(n_estimators=120,random_state=101)
dt=DecisionTreeClassifier()
gnb=GaussianNB()

voting_Hard=VotingClassifier(estimators=[("GNB",gnb),("RF",rf),("DT",dt),("KNN_7",knn_7)],voting="hard")


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