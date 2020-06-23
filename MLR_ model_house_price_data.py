#Name: Mohan Subhash Paliwal..
#ID: AIS035
#------------------------------------------------------------------------------
# data set 1 :- House price prediction in USA
# -----rread data------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:\\Users\\admin\\Desktop\\housepp.csv")
print(data.head())

#-------------------------------------------------
#----- check information ---------------
data.info()

#------------------------------------------------
#------describe data-------------------------------
print(data.describe())

#--------------------------------------------------
#---plot pair plot  and observ-----------------------

#sns.pairplot(data[["price","bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","sqft_above","sqft_basement","yr_built"]])
#plt.show()

# in pair plot we observ that some variabes are not highly correlated with price of house
sns.heatmap(data.corr())
plt.show()
# from correlation matrix plot we select x variable sqrt_living for SLR model_1

sns.distplot(data["price"])
plt.show()

#---------seperate data for model 1 x var=sqrt living-------------------------
#--------using x=srft living_only--------------
x=data.iloc[:,4].values.reshape(-1,1)
y=data.iloc[:,1].values.reshape(-1,1)
#print y.head()

#------traon test split----------------------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)

#----------model import------------------------------
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
#--------fit model----------------------------------
lm.fit(x_train,y_train)
print(lm.intercept_)
print(lm.coef_)

#-------predict data test-----------------------------
y_pred=lm.predict(x_test)
print(y_pred)
plt.scatter(y_test,y_pred)
plt.show()

#---------------------------------------------------------------------------
#---------seperate data for model 2 x var=sqrt living, sqrt above,bedrooms -------------------------
#--------using x=srft living_only--------------
x2=data.iloc[:,[4,2,10]]
y2=data.iloc[:,1]
#print y.head()

#------traon test split----------------------------
from sklearn.model_selection import train_test_split

x_train2,x_test2,y_train2,y_test2=train_test_split(x2,y2,test_size=0.20,random_state=101)
print("x_train2 :",x_train2.shape)
print("y_train2:",y_train2.shape)
print("x_test2:",x_test2.shape)
print("y_test2:",y_test2.shape)
#----------model import------------------------------
from sklearn.linear_model import LinearRegression
lm2=LinearRegression()
#--------fit model----------------------------------
lm2.fit(x_train2,y_train2)
print(lm2.intercept_)
print(lm2.coef_)

#-------predict data test-----------------------------
y_pred2=lm2.predict(x_test2)
#print(y_pred2)
plt.scatter(y_test2,y_pred2)
#plt.show()
#----------------------------------------------------------------------------------------
#---------seperate data for model 3 x var=sqrt living, sqrt above,bedrooms,bathrooms,views-------------------------
#--------using x=srft living_only--------------
x3=data.iloc[:,[4,2,10,3,8]]
y3=data.iloc[:,1]
#print y.head()

#------traon test split----------------------------
from sklearn.model_selection import train_test_split

x_train3,x_test3,y_train3,y_test3=train_test_split(x3,y3,test_size=0.20,random_state=101)
print("x_train3 :",x_train3.shape)
print("y_train3:",y_train3.shape)
print("x_test3:",x_test3.shape)
print("y_test3:",y_test3.shape)

#----------model import------------------------------
from sklearn.linear_model import LinearRegression
lm3=LinearRegression()

#--------fit model----------------------------------
lm3.fit(x_train3,y_train3)
print(lm3.intercept_)
print(lm3.coef_)

#-------predict data test-----------------------------
y_pred3=lm3.predict(x_test3)
#print(y_pred3)

plt.scatter(y_test3,y_pred3)
#plt.show()

#-----------------------------------------------------------------------------------------
# Mean Squared Error
#---------------------------------------------------
from sklearn import metrics

MAE_1=metrics.mean_absolute_error(y_test,y_pred)
MSE_1=metrics.mean_squared_error(y_test,y_pred)
RMSE_1=np.sqrt(metrics.mean_squared_error(y_test,y_pred))

MAE_2=metrics.mean_absolute_error(y_test2,y_pred2)
MSE_2=metrics.mean_squared_error(y_test2,y_pred2)
RMSE_2=np.sqrt(metrics.mean_squared_error(y_test2,y_pred2))

MAE_3=metrics.mean_absolute_error(y_test3,y_pred3)
MSE_3=metrics.mean_squared_error(y_test3,y_pred3)
RMSE_3=np.sqrt(metrics.mean_squared_error(y_test3,y_pred3))

print(" MSE 1:> ",MSE_1,"   MSE_2 :>  ",MSE_2,"  MSE_3:>  ",MSE_3)
print(" MAE 1:> ",MSE_1,"   MAE_2 :>  ",MAE_2,"  MSE_3:>  ",MAE_3)
print(" RMSE 1:> ",RMSE_1,"   RMSE_2 :>  ",RMSE_2,"  RMSE_3:>  ",RMSE_3)

print(min(MSE_1,MSE_2,MSE_3))
print(min(MAE_1,MAE_2,MAE_3))
print(min(RMSE_1,RMSE_2,RMSE_3))

#--using minimum MSE and minimum MAE criteria we select model 3 for prediction
#using independat variables sqrt living, sqrt above,bedrooms,bathrooms,views
#-----------------------------------------------------------
