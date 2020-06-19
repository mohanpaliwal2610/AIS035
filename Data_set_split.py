#Name: Mohan Subhash Paliwal..
#ID: AIS035
#------------------------------------------------------------------------------
# data set 1 :- heart (kaggle.com)
import pandas as pd
heart=pd.read_csv("C:\\Users\\admin\\Desktop\\AIS Solutions\\New folder\\Data_sets\\33180_43520_bundle_archive\\heart.csv")
print heart.head()
x=heart.iloc[:,0:13]
y=heart.iloc[:,13:14]
print y.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)


#------------------------------------------------------------------------------
# data set 1 :- diabetes (kaggle.com)
import pandas as pd
heart=pd.read_csv("C:\\Users\\admin\\Desktop\\AIS Solutions\\New folder\\Data_sets\\228_482_bundle_archive\\diabetes.csv")
print heart.head()
x=heart.iloc[:,0:8]
y=heart.iloc[:,8:9]
print y.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)


#------------------------------------------------------------------------------
# data set 3 :- insurance (kaggle.com)
import pandas as pd
heart=pd.read_csv("C:\\Users\\admin\\Desktop\\AIS Solutions\\New folder\\Data_sets\\13720_18513_bundle_archive\\insurance.csv")
print heart.head()
x=heart.iloc[:,0:6]
y=heart.iloc[:,6:7]
print y.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)




