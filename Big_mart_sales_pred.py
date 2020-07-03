#Name: Mohan Subhash Paliwal..
#ID: AIS035
"""
# === data name: Sales Prediction for Big Mart Outlets

The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities.
Also, certain attributes of each product and store have been defined. The aim is to build a predictive model
and predict the sales of each product at a particular outlet.
Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales.
Please note that the data may have missing values as some stores might not report all the data due to technical glitches.
 Hence, it will be required to treat them accordingly.

#======Data Dictionary

We have train (8523) and test (5681) data set, train data set has both input and output variable(s).
 You need to predict the sales for test data set.
Train file: CSV containing the item outlet information with sales value
Variable         |	Description
Item_Identifier:-	Unique product ID
Item_Weight  :-    	Weight of product
Item_Fat_Content :-	Whether the product is low fat or not
Item_Visibility :-	The % of total display area of all products in a store allocated to the particular product
Item_Type 	:-      The category to which the product belongs
Item_MRP 	:-       Maximum Retail Price (list price) of the product
Outlet_Identifier:-	Unique store ID
Outlet_Establishment_Year:- 	The year in which store was established
Outlet_Size :-	The size of the store in terms of ground area covered
Outlet_Location_Type :-	The type of city in which the store is located
Outlet_Type 	:-Whether the outlet is just a grocery store or some sort of supermarket
Item_Outlet_Sales :-	Sales of the product in the particular store. This is the outcome variable to be predicted.


"""
#------------------------------------------------------------------------------
# data set 1 :- Big mart sales prediction
# -----rread data------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------
sales=pd.read_csv("C:\\Users\\admin\\Desktop\\BigMartSalesProduction\\train_v9rqX0R.csv")
print(sales.head())

#------------------------------------------------------

#Check the head of customers, and check out its info() and describe() methods.

print(sales.info())
print(sales.describe())

'''
 Item_Identifier  Item_Weight  ...        Outlet_Type  Item_Outlet_Sales
0           FDA15         9.30  ...  Supermarket Type1          3735.1380
1           DRC01         5.92  ...  Supermarket Type2           443.4228
2           FDN15        17.50  ...  Supermarket Type1          2097.2700
3           FDX07        19.20  ...      Grocery Store           732.3800
4           NCD19         8.93  ...  Supermarket Type1           994.7052

[5 rows x 12 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8523 entries, 0 to 8522
Data columns (total 12 columns):
Item_Identifier              8523 non-null object
Item_Weight                  7060 non-null float64
Item_Fat_Content             8523 non-null object
Item_Visibility              8523 non-null float64
Item_Type                    8523 non-null object
Item_MRP                     8523 non-null float64
Outlet_Identifier            8523 non-null object
Outlet_Establishment_Year    8523 non-null int64
Outlet_Size                  6113 non-null object
Outlet_Location_Type         8523 non-null object
Outlet_Type                  8523 non-null object
Item_Outlet_Sales            8523 non-null float64
dtypes: float64(4), int64(1), object(7)
memory usage: 566.0+ KB
None
       Item_Weight  ...  Item_Outlet_Sales
count  7060.000000  ...        8523.000000
mean     12.857645  ...        2181.288914
std       4.643456  ...        1706.499616
min       4.555000  ...          33.290000
25%       8.773750  ...         834.247400
50%      12.600000  ...        1794.331000
75%      16.850000  ...        3101.296400
max      21.350000  ...       13086.964800
'''
#===============manipulation and cleaning of data=======
sales.drop("Item_Identifier",axis=1,inplace=True)
sales.drop("Outlet_Identifier",axis=1,inplace=True)
sales.drop("Outlet_Establishment_Year",axis=1,inplace=True)
print sales.head()
#-------------------------------------------------------------------------
print sales.corr()

f,ax=plt.subplots(figsize=(9,8))
sns.heatmap(sales.corr(),ax=ax,cmap="YlGnBu",linewidths=0.1)
plt.show()

sns.boxplot(sales["Item_Weight"])
plt.show()


# seperating categorical variable and using dummy variables
sales_c=sales.loc[:,["Item_Fat_Content","Item_Type","Outlet_Size","Outlet_Location_Type","Outlet_Type"]]
print(sales_c.head())

#-----fiilling outlet size by median
sales_c["Outlet_Size"].fillna("Medium",inplace= True)
print(sales_c["Outlet_Size"].value_counts())
sales_c["Outlet_Size"].isnull().sum()


dum_sales_c=pd.get_dummies(sales_c,drop_first=True)
print(dum_sales_c.head())

#--seperating numerical variable
sales_n=sales.loc[:,["Item_Weight","Item_Visibility","Item_MRP","Item_Outlet_Sales"]]
sales_n.isnull().sum()
sales_n["Item_Weight"].fillna(sales_n["Item_Weight"].mean(),inplace=True)
sales_n.isnull().sum()

#concating numerical and cateegoical data
sales_1=pd.concat([sales_n,dum_sales_c],axis=1)
print(sales_1.head())
print(sales_1.columns)
print sales_1
#------Model Building------------------

#---------seperate data--------------------------
x=sales_1.loc[:,['Item_Weight', 'Item_Visibility', 'Item_MRP','Item_Fat_Content_Low Fat', 'Item_Fat_Content_Regular',
       'Item_Fat_Content_low fat','Item_Fat_Content_reg',
       'Item_Type_Breads','Item_Type_Breakfast','Item_Type_Canned',
       'Item_Type_Dairy','Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables','Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene','Item_Type_Household',
       'Item_Type_Meat','Item_Type_Others','Item_Type_Seafood',
       'Item_Type_Snack Foods','Item_Type_Soft Drinks',
       'Item_Type_Starchy Foods','Outlet_Size_Medium','Outlet_Size_Small',
       'Outlet_Location_Type_Tier 2','Outlet_Location_Type_Tier 3',
       'Outlet_Type_Supermarket Type1','Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3']]
y=sales_1.loc[:,['Item_Outlet_Sales']].values.reshape(-1,1)


#print y.head()
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
#from sklearn.linear_model import LinearRegression
#lm=LinearRegression()

param={'n_estimators':500,'max_depth':100}
from sklearn.ensemble import RandomForestRegressor
lm=RandomForestRegressor(**param)


#--------fit model----------------------------------
lm.fit(x_train,y_train)


#------------------------------------------------------
#Predicting Test Data
#Now that we have fit our model, let's evaluate its performance by predicting off the test values!
#** Use lm.predict() to predict off the X_test set of the data

#-------predict data test-----------------------------
y_pred=lm.predict(x_test)
print type(y_pred)
#print y_pred.reshape(-1,1)
print(y_pred)
plt.scatter(y_test,y_pred)
plt.show()
print(min(y_pred))



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


"""=========OUTPUT============
[1685.3874776 3618.7242016 1338.0036644 ... 1195.7341888 2600.887778
 2313.9173252]
45.62727400000006
0.5500409410569771
(' MSE :> ', 1208866.2217167409)
(' MAE :> ', 772.0687879422761)
(' RMSE :> ', 1099.4845254557888)

=============================="""

#print sales["Item_Identifier"].head()

#=================================================================================================

#------------------------------------------------------
sales=pd.read_csv("C:\\Users\\admin\\Desktop\\BigMartSalesProduction\\test_AbJTz2l.csv")
print(sales.head())

#------------------------------------------------------

#Check the head of customers, and check out its info() and describe() methods.

#-------------------------------------------------------------------------

sales.drop("Item_Identifier",axis=1,inplace=True)
sales.drop("Outlet_Identifier",axis=1,inplace=True)
sales.drop("Outlet_Establishment_Year",axis=1,inplace=True)

'''
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(sales["Item_Identifier"])

sales["Item_Identifier"]=le.transform(sales["Item_Identifier"])

le.fit(sales["Outlet_Identifier"])
sales["Outlet_Identifier"]=le.transform(sales["Outlet_Identifier"])
'''

# seperating categorical variable and using dummy variables
sales_c=sales.loc[:,["Item_Fat_Content","Item_Type","Outlet_Size","Outlet_Location_Type","Outlet_Type"]]
print(sales_c.head())

#-----fiilling outlet size by median
sales_c["Outlet_Size"].fillna("Medium",inplace= True)
print(sales_c["Outlet_Size"].value_counts())
sales_c["Outlet_Size"].isnull().sum()


dum_sales_c=pd.get_dummies(sales_c,drop_first=True)
print(dum_sales_c.head())

#--seperating numerical variable
sales_n=sales.loc[:,["Item_Weight","Item_Visibility","Item_MRP"]]
sales_n.isnull().sum()
sales_n["Item_Weight"].fillna(sales_n["Item_Weight"].mean(),inplace=True)
sales_n.isnull().sum()

#concating numerical and cateegoical data
sales_1=pd.concat([sales_n,dum_sales_c],axis=1)
print(sales_1.head())
print(sales_1.columns)
print sales_1
#------Model Building------------------

#---------seperate data--------------------------
x=sales_1.loc[:,['Item_Weight', 'Item_Visibility', 'Item_MRP','Item_Fat_Content_Low Fat', 'Item_Fat_Content_Regular',
       'Item_Fat_Content_low fat','Item_Fat_Content_reg',
       'Item_Type_Breads','Item_Type_Breakfast','Item_Type_Canned',
       'Item_Type_Dairy','Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables','Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene','Item_Type_Household',
       'Item_Type_Meat','Item_Type_Others','Item_Type_Seafood',
       'Item_Type_Snack Foods','Item_Type_Soft Drinks',
       'Item_Type_Starchy Foods','Outlet_Size_Medium','Outlet_Size_Small',
       'Outlet_Location_Type_Tier 2','Outlet_Location_Type_Tier 3',
       'Outlet_Type_Supermarket Type1','Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3']]


#Training the Model
#Import LinearRegression from sklearn.linear_model
#----------model import------------------------------
#from sklearn.linear_model import LinearRegression
#lm=LinearRegression()


#------------------------------------------------------
#Predicting Test Data

#-------predict data test-----------------------------
y_pred=lm.predict(x)
print(y_pred)

#Generating and submotting CSV file --------------------------------


sales_1=pd.read_csv("C:\\Users\\admin\\Desktop\\BigMartSalesProduction\\test_AbJTz2l.csv")

Item_Identifier=sales_1["Item_Identifier"]
Outlet_Identifier=sales_1["Outlet_Identifier"]



submission=pd.DataFrame({"Item_Identifier":Item_Identifier,"Item_Outlet_Sales":y_pred,"Outlet_Identifier":Outlet_Identifier})
#yd={"predict":y_pred}
#submission=pd.DataFrame(yd)

submission.to_csv("C:\\Users\\admin\\Desktop\\BigMartSalesProduction\\sample_submission13.csv",index=False)

