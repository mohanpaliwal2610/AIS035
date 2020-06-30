#Name: Mohan Subhash Paliwal..
#ID: AIS035
#------------------------------------------------------------------------------
# data set 1 :- loan status prediction data
# -----rread data------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------
data=pd.read_csv("C:\\Users\\admin\\Desktop\\Loan_data\\train_loan.csv")
print(data.head())

#------------------------------------------------------

#Check the head of customers, and check out its info() and describe() methods.

print(data.info())
print(data.describe())
#==================================================
data.drop("Loan_ID",axis=1,inplace=True)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(data["Loan_Status"])
data["Loan_Status"]=le.transform(data["Loan_Status"])
print data.head()

print(data["Gender"].value_counts())
print(data["Gender"].isnull().sum())
data["Gender"].fillna("Male",inplace=True)
print(data["Gender"].isnull().sum())

print(data["Married"].value_counts())
data["Married"].fillna("Yes",inplace=True)
print(data["Gender"].isnull().sum())

print(data["Dependents"].value_counts())
data["Dependents"].fillna("0",inplace=True)
print(data["Dependents"].isnull().sum())

print(data["Self_Employed"].value_counts())
data["Self_Employed"].fillna("No",inplace=True)
print(data["Self_Employed"].isnull().sum())

data["LoanAmount"].hist(bins=20)
plt.show()

print data["LoanAmount"].median()
data["LoanAmount"].fillna(data["LoanAmount"].median(),inplace=True)
print(data["LoanAmount"].isnull().sum())

data["Loan_Amount_Term"].hist(bins=20)
plt.show()

print data["Loan_Amount_Term"].median()
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median(),inplace=True)
print(data["Loan_Amount_Term"].isnull().sum())

print(data["Credit_History"].value_counts())
data["Credit_History"].fillna(1,inplace=True)
print(data["Credit_History"].isnull().sum())

print(data["Credit_History"].value_counts())

#==================================================
# categrical variables
data_c=data.loc[:,["Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area"]]
print(data_c.head())


dum_data_c=pd.get_dummies(data_c,drop_first=False)
print(dum_data_c.head())
print dum_data_c.isnull().sum()

# numerical variables
data_n=data.loc[:,["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]]
print(data_c.head())

##concating numerical and cateegoical data
data_1=pd.concat([data_n,dum_data_c],axis=1)
print(data_1.head())
print(data_1.columns)

#------Model Building------------------

#---------seperate data--------------------------
x=data_1.loc[:,['ApplicantIncome','CoapplicantIncome','LoanAmount',
       'Loan_Amount_Term','Credit_History','Gender_Female',
       'Gender_Male','Married_No','Married_Yes','Dependents_0',
       'Dependents_1','Dependents_2','Dependents_3+',
       'Education_Graduate','Education_Not Graduate','Self_Employed_No',
       'Self_Employed_Yes','Property_Area_Rural',
       'Property_Area_Semiurban','Property_Area_Urban']]
y=data.loc[:,['Loan_Status']].values.reshape(-1,1)
y=y.ravel()


#print y.head()
#------traon test split----------------------------

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=300)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)

#================================================
#model fitting------------------

#----------model import---KNeighborsClassifier---------------------------
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=12)                # Accuracy 69.189189
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


'''
#==========
#============Naive Bayes Classification ==using GaussianNB
from sklearn.naive_bayes import GaussianNB              # accuracy 83.7837
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
#=============Descriminant Analysis classification==========================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
gm=LinearDiscriminantAnalysis()                                 # accuracy 84.3243
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
#gm=DecisionTreeClassifier()                                  #accuracy 71.8991 using ginni index
gm=DecisionTreeClassifier(criterion="entropy",max_depth=1000)   # accuracy 68.6486 using entropy
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
#-------------------------------------------------------------------------------------------
#for test dataprediction
data=pd.read_csv("C:\\Users\\admin\\Desktop\\Loan_data\\test_loan.csv")

#------------------------------------------------------

#Check the head of customers, and check out its info() and describe() methods.

#==================================================
#data.drop("Loan_ID",axis=1,inplace=True)

print(data["Gender"].value_counts())
print(data["Gender"].isnull().sum())
data["Gender"].fillna("Male",inplace=True)
print(data["Gender"].isnull().sum())

print(data["Married"].value_counts())
data["Married"].fillna("Yes",inplace=True)
print(data["Gender"].isnull().sum())

print(data["Dependents"].value_counts())
data["Dependents"].fillna("0",inplace=True)
print(data["Dependents"].isnull().sum())

print(data["Self_Employed"].value_counts())
data["Self_Employed"].fillna("No",inplace=True)
print(data["Self_Employed"].isnull().sum())

data["LoanAmount"].hist(bins=20)
plt.show()

print data["LoanAmount"].median()
data["LoanAmount"].fillna(data["LoanAmount"].median(),inplace=True)
print(data["LoanAmount"].isnull().sum())

data["Loan_Amount_Term"].hist(bins=20)
plt.show()

print data["Loan_Amount_Term"].median()
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median(),inplace=True)
print(data["Loan_Amount_Term"].isnull().sum())

print(data["Credit_History"].value_counts())
data["Credit_History"].fillna(1,inplace=True)
print(data["Credit_History"].isnull().sum())
print(data["Credit_History"].value_counts())

#==================================================
# categrical variables
data_c=data.loc[:,["Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area"]]
dum_data_c=pd.get_dummies(data_c,drop_first=False)

# numerical variables
data_n=data.loc[:,["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]]
print(data_c.head())

##concating numerical and cateegoical data
data_1=pd.concat([data_n,dum_data_c],axis=1)

x=data_1.loc[:,['ApplicantIncome','CoapplicantIncome','LoanAmount',
       'Loan_Amount_Term','Credit_History','Gender_Female',
       'Gender_Male','Married_No','Married_Yes','Dependents_0',
       'Dependents_1','Dependents_2','Dependents_3+',
       'Education_Graduate','Education_Not Graduate','Self_Employed_No',
       'Self_Employed_Yes','Property_Area_Rural',
       'Property_Area_Semiurban','Property_Area_Urban']]
y_pred=gm.predict(x)
'''
'''
y_pred=np.array(y_pred)
for i in range(0,len(y_pred)):
    if y_pred[i]==1:
        y_pred[i]="Y"
    else:
        y_pred[i]="N"
'''
'''
print(y_pred)
print(len(y_pred))
print(type(y_pred))

submission=pd.DataFrame({"Loan_ID":data["Loan_ID"],"Loan_status":y_pred})

submission.to_csv("C:\\Users\\admin\\Desktop\\Loan_data\\sample_submission1.csv",index=False)

'''