#Name: Mohan Subhash Paliwal..
#ID: AIS035

#Python Exercise No: 3
#Subject: List and Tuple in python
#------------------------------------------------------------
#Q1) What is tuple?
'''
Ans:-Immutable data type-can't update / change
Ordered Sequences- index starting from 0
Declared using ()
Each value is seperated by comma
'''
#-----------------------------------------------------------
#Q2)What is difference between list and tuple?
'''
The differences between tuples and lists are, the tuples cannot be changed unlike lists
and tuples use parentheses, whereas lists use square brackets
'''
#-----------------------------------------------------------
#Q3) What is the result of range (1, 12, 2)?
x=range(1,12,2)
print("result of range(1,12,2) is : ", x)

#------------------------------------------------------------
#Q4) Assuming a= [10, 20, 30, 40, 50], Find length of a list?  Find maximum and minimum number of a list?
a=[10,20,30,40,50]
print("the lenngth of ",a,"is ",len(a))
print("maximum of",a,"is",max(a))
print("minimum of",a,"is",min(a))

#-----------------------------------------------------------
#Q5) If x=(1,2,3,4,5,1,3,1) then what is the result of x*2 and x.count(1)?
x=(1,2,3,4,5,1,3,1)
print(x*2)
print(x.count(1))

#-----------------------------------------------------------
#Q6) Create a list of colors name and then print first and last color name of the list using slicing?

col=["green","white","black","red","blue","yello","orange"]
print("1st name of color in the list is:",col[0])
print("last name of color in the list is:",col[-1])

#-----------------------------------------------------------
#Q7) If list = [1,4,7,'a','c',"rat","mat",4,6,4] then add element 12 using append method and
#remove rat element using remove method in the given list?

list=[1,4,7,'a','c',"rat","mat",4,6,4]
print(list)
list.append(12)
print(list)
list.remove("rat")
print(list)

#---------------------------------------------------------------------
#Q8)Create two list and concatenate them and then repeat 1st list 3 times?

list1=[1,2,3,4,5]
list2=[6,"seven",8,"nine",10]
list1.extend(list2)
print(list1)

#---------------------------------------------------------------------
#Q9)Create a tuple of fruits name and then access first 4 elements from the tuple?
tup1=("mango","banana","apple","watermelon","orange","lime","papaya""lemon")
print(tup1[0:4])

#----------------------------------------------------------------------
#Q10) Create a list using range function and then convert this list into tuple using tuple function?

ls=range(1,21,2)
print(ls)
print(type(ls))
tp=tuple(ls)
print(tp)
print(type(tp))

#-------------------------------------------------------------------------
#Q11) Create a tuple using eval function and then obtained the length of tuple?
x=tuple(eval("[1,2,3,4,5]"))
print(type(x))
print(len(x))



#---------------------------------------------------------------------------
#Q12) Create a tuple of integer number and then sort the elements of tuple?

tp12=(1,5,2,7,3,12,4,40,50,10,56)
print(sorted(tp12))

#-----------------------------------------------------------------------
#Q13)Create a list of colors name and then insert purple color on 3rd position using insert method?
col=["green","white","black","red","blue","yello","orange"]
col.insert(2,"purple")
print(col)

#---------------------------------------------------------------------
#Q14) If x= (50, 60, 70, 80, 90, (200,100)) then access 200,100 from the tuple?
x=(50,60,70,80,90,(200,100))
print(x[5])

#------------------------------------------------------------------------
#Q15Create a nested tuple and nested list?
ntp=(1,2,(3,4),(5,6,(7,8),9),10)
print(ntp)
nlst=[11,22,[33,44,[555,888,999,[420,143],111]]]
print(nlst)

#------------------------------------------------------------------------------------
