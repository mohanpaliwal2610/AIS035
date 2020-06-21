#Name: Mohan Subhash Paliwal..
#ID: AIS035
#------------------------------------------------------------------------------
# data set 1 :- House price prediction in USA
# -----rread data------------------------
import pandas as pd
data=pd.read_csv("C:\\Users\\admin\\Desktop\\housepp.csv")
print(data.head())
#---------seperate data--------------------------
x=data.iloc[:,2:14]
y=data.iloc[:,1:2]
print y.head()
#------traon test split----------------------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)
#----------model import------------------------------
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=5,weights="uniform",algorithm="auto")

#--------fit model----------------------------------
knn.fit(x_train,y_train)

#-------predict data test-----------------------------
y_pred=knn.predict(x_test)
print(y_pred)


#---------------------------------------------------
# Mean Squared Error
#---------------------------------------------------
from sklearn.metrics import mean_squared_error
import numpy as np
print(len(data))
rk=range(1,50)
t=[]
for i in rk:
    knn=KNeighborsRegressor(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    MSE=np.sqrt(mean_squared_error(y_test,y_pred))
    t.append(float(MSE))
    print("MSE when k is "+str(i)+"= ",MSE)
print(min(t))
x=t.index(min(t))
print(x)

print(" the minimum MSE are found with value of K is :->",rk[x])
#-----------------------------------------------------
# Root Mean square error
#------------------------------------------------------
from sklearn.metrics import mean_squared_error
import numpy as np
import math
t=[]
rk=range(1,50)
for i in rk:
    knn=KNeighborsRegressor(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    MSE=np.sqrt(mean_squared_error(y_test,y_pred))
    RMSE=math.sqrt(MSE)
    t.append(float(RMSE))
    print("RMSE when k is "+str(i)+"= ",RMSE)

print(min(t))
x = t.index(min(t))
print(x)

print(" the minimum RMSE are found with value of K is :->", rk[x])

#-----------------------------------------------------
# Mean Absolute error
#------------------------------------------------------
from sklearn.metrics import mean_absolute_error
rk=range(1,50)
t=[]
for i in rk:
    knn=KNeighborsRegressor(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    MAE=mean_absolute_error(y_test,y_pred)
    t.append(float(MAE))
    print("MAE when k is "+str(i)+"=",MAE)

print(min(t))
x = t.index(min(t))
print(x)

print(" the minimum MAE are found with value of K is :->", rk[x])

    #-----------------------------------------------------------