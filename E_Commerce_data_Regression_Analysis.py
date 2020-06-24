#Name: Mohan Subhash Paliwal..
#ID: AIS035
#------------------------------------------------------------------------------
# data set 1 :- E Commerce Regression Analysis  project exercise
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

#--------------------------------------------------------
#Exploratory Data Analysis
#Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns.
#Does the correlation make sense?

sns.jointplot(customers["Time on Website"],customers["Yearly Amount Spent"])
plt.show()
#__here we observe that there is no correlation between 'time on webside' and 'yrearly amout spent'

sns.jointplot(customers["Time on App"],customers["Yearly Amount Spent"])
plt.show()
#__here we observe that there is some positive correlation between 'time on app' and 'yrearly amout spent'


#-------------------------------------------------------
#Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership

sns.jointplot(customers["Time on App"],customers["Length of Membership"],kind="hex")
plt.show()

sns.pairplot(customers)
plt.show()

#------------------------------------------------------
#Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membershi
from seaborn import lmplot
#sns.scatterplot(customers["Yearly Amount Spent"],customers["Length of Membership"],palette="True")
lmplot(x="Yearly Amount Spent",y="Length of Membership",data=customers)
plt.show()

#--------------------------------------------------
#Training and Testing Data
#Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets. ** Set a
#variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount
#Spent" column.

#---------seperate data--------------------------
x=customers.iloc[:,3:7]
y=customers.iloc[:,7]
print y.head()
#------traon test split----------------------------
#Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set
#test_size=0.3 and random_state=101

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

#------------------------------------------------------
#Predicting Test Data
#Now that we have fit our model, let's evaluate its performance by predicting off the test values!
#** Use lm.predict() to predict off the X_test set of the data

#-------predict data test-----------------------------
y_pred=lm.predict(x_test)
print(y_pred)
plt.scatter(y_test,y_pred)
plt.show()

#Evaluating the Model
#Let's evaluate our model performance by calculating the residual sum of squares and the explained variance
#score (R^2)

from sklearn import metrics
print(metrics.r2_score(y_test,y_pred))

#Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Erro

MAE=metrics.mean_absolute_error(y_test,y_pred)
MSE=metrics.mean_squared_error(y_test,y_pred)
RMSE=np.sqrt(metrics.mean_squared_error(y_test,y_pred))

print(" MSE :> ",MSE)
print(" MAE :> ",MAE)
print(" RMSE :> ",RMSE)

#---------------------------------------------
#Residuals

rs=y_test-y_pred
sns.distplot(rs,bins=50)
plt.show()

'''
Conclusion
We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website
development? Or maybe that doesn't even really matter, and Membership Time is what is really important.
Let's see if we can interpret the coefficients at all to get an idea.
'''
coeff=pd.DataFrame(lm.coef_,x.columns,columns=["coefficient"])
print(coeff)

"""
Avg. Session Length     25.981550
Time on App             38.590159
Time on Website          0.190405
Length of Membership    61.279097
"""
"""
if one unit increse in Average session of in-store style advice sessions then yearly amount spent by costomer increse 25.9815 times.
if one unit increse in time on app then yearly amount spent by costomer increse 38.590159 times.
if one unit increse in time on website then yearly amount spent by costomer increse 0.1904 times.
if one unit increse in tLength of Membership then yearly amount spent by costomer increse 61.279097 times.

"""
#Do you think the company should focus more on their mobile app or on their website?

#yes company should focus more on there mobile app for better performance than there webside.
