#Name: Mohan Subhash Paliwal..
#ID: AIS035
#------------------------------------------------------------------------------
# data set 1 :- E Commerce visualization and analysis
# -----rread data------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------
#Read in the Ecommerce Customers csv file as a DataFrame called customers.

customers=pd.read_csv("C:\\Users\\admin\\Desktop\\AIS Solutions\\Ecommerce Customers.csv")
print(customers.head())

#------------------------------------------------------
#Check the head of customers, and check out its info() and describe() methods.

print(customers.info())
print(customers.describe())

#-------------------------------------------------------------------------

sns.jointplot(customers["Time on Website"],customers["Yearly Amount Spent"])
plt.show()
#__here we observe that there is no correlation between 'time on webside' and 'yrearly amout spent'

sns.jointplot(customers["Time on App"],customers["Yearly Amount Spent"])
plt.show()
#__here we observe that there is some positive correlation between 'time on app' and 'yrearly amout spent'

sns.distplot(customers["Yearly Amount Spent"],kde=False,rug=True)
plt.show()
# from above chart we observe that yearly amount spend are normaly distributed


sns.boxplot(customers["Yearly Amount Spent"])
plt.show()
#from box plot we observe that many outliers are available

sns.jointplot(customers["Time on App"],customers["Time on Website"])
plt.show()
# time on app and time on webside are not associated

sns.regplot(x="Time on App",y="Yearly Amount Spent",data=customers[customers["Time on App"]>12])
plt.show()


customers["Yearly Amount Spent"].hist()
plt.show()
#yearly amount spend variable are normaly distributed


plt.style.use("ggplot")
customers["Yearly Amount Spent"].hist()
plt.show()



customers["Yearly Amount Spent"].plot.area(alpha=0.3)
plt.show()



customers.plot.scatter(x="Time on App",y="Yearly Amount Spent")
plt.show()
# time on app are positively correlated with yearly smount spend

customers.plot.scatter(x="Time on App",y="Yearly Amount Spent",c="Time on Website",Cmap="rainbow")
plt.show()


customers.plot.hexbin(x="Time on App",y="Yearly Amount Spent")
plt.show()




#find correlation of all variables
print(customers.corr())

# from the above correlation matrix we find that time on websid are not correlated with yearly spend amound by customer
#length of membership are highly correlated with yearly spend amound by customer
#------------------------------------------
#variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount
#Spent" column.
# independant variables->Time on App, Avg. Session Length, Length of Membership
#---------seperate data--------------------------
x=customers.iloc[:,[3,4,6]]
y=customers.iloc[:,7]
print y.head()
#------traon test split----------------------------

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=101)
print("x_train :",x_train.shape)
print("y_train :",y_train.shape)
print("x_test :",x_test.shape)
print("y_test :",y_test.shape)

#--------------------------------------------
#Training the Model
#Import LinearRegression from sklearn.linear_model
#----------model import------------------------------
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
#--------fit model----------------------------------
lm.fit(x_train,y_train)

#Print out the coefficients of the model
print(lm.intercept_)
print(lm.coef_)


#-------predict data test-----------------------------
y_pred=lm.predict(x_test)
print(y_pred)
plt.scatter(y_test,y_pred)
plt.show()

#score (R^2)

from sklearn import metrics
print(metrics.r2_score(y_test,y_pred))
# this model are r2=0.9889519444175247 model are 89.89 % good model

#-------------------------------------------------------------------
print(customers["Yearly Amount Spent"].describe())
'''
count    500.000000
mean     499.314038
std       79.314782
min      256.670582
25%      445.038277
50%      498.887875
75%      549.313828
max      765.518462
# peoples spend 499.31 average amound yearly with 79 SD
 75% peoples spend up to 549.31 amount yesrly
 maximum amount spend by customers is 765 yearly and minimum amount spend by customer is 256.67.
'''



# my intrest is find how many people spend more than 500 rs
print(customers[customers["Yearly Amount Spent"]>=500].count())
# 249 peoples are spend more than 500 rs yearly.


