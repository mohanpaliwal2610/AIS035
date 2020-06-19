#Name: Mohan Subhash Paliwal..
#ID: AIS035

#Python Exercise No: 2
#Subject: Arrays and String in python
#-------------------------------------------------------------------
#Q1) What is an array?
'''
ans:-Array is a container which can hold a fix no of items and these items should be of the same type.
If you creat arrays using the array module all the element of the array must be of same numeric type.

'''
#---------------------------------------------------------------------
#Q2) . What is difference between array and list?
'''
ans:- Arrays need to be declared where as lists do not need to declaration because they are part of python's syntax.
lists are more often used than array.
Array are more efficient than lists for some uses
list items are enclosed in square bracket [],list are mutable, ordered,do not need to be unique.dublicate is possible.
in array before use we import numpy package. like numpy.array(), array are great for numerical operations.

'''
#-----------------------------------------------------------------------
#Q3) How to change the shape of an array?
'''
we use reshape(no of row, no of column) command for reshape array.
e.g.:
import numpy as np
x=np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
print x.reshape(6,2)
'''
#--------------------------------------------------------------------------
#Q4) What does the name.strip () do?
'''
name.strip() remove the blank spaces of both side of name variable
e.g.: name="     hello india     "
print(name.strip())
 name="hello india"
'''
#--------------------------------------------------------------------------
#Q5) When we use triple single or triple double quotes to define a string?
'''
when we write multiline comment ot multiline string then we use triple single or double quotes.
'''
#-------------------------------------------------------------------------
#Q6)Create a character type array?
import numpy as np
arr=np.array(["A","B","C","D"])
print(arr)

#--------------------------------------------------------------------------
#Q7) Create a hello string and access the string character "ll" in python?

str1='hello'
print(str1[2:4])

#---------------------------------------------------------------------------
#Q8). Create an array to get largest and smallest number using minimum and maximum function?
import numpy as np
arr=np.array([1,2,3,4,5,6,7,8,9,10])
print(np.min(arr))
print(np.max(arr))

#------------------------------------------------------------------------------
#Q9)If name is core python th en access the python from name and reverse python using slicing?
name="core python"
new=name[5:11]
print(new)
print(new[::-1])

#--------------------------------------------------------------------------------
#Q10) Create an array of integer type and add element 12 using append method and remove one element using remove methods?
import numpy as np
arr=np.array([1,2,3,4,5,6,7,8,9,10])
print(np.append(arr,12))

#--------------------------------------------------------------------------------
#Q11) Create a single dimensional and 3 dimensional array using numpy?
import numpy as np
arr1=np.array([1,2,4,5,5])
print(arr1)
arr3=np.array([[[1,2],[3,4]],[[6,7],[8,9]]])    #(2,2,2)
print(arr3)
#----------------------------------------------------------------------------------
#Q12) Create a Programiz string and convert the string into upper, lower case? And replace
#given string by program using replace method?
str1="Programiz"
print(str1.upper())
print(str1.lower())
str2=str1.replace("Programiz","program")
print(str2)
#--------------------------------------------------------------------------------
#Q13) Create a two array one using linspace and another using arange function and then subtract array1 from array 2?

a1=np.linspace(1,20,20)
print a1
a2=np.arange(21,41)
print a2
print(a2-a1)
#--------------------------------------------------------------------------------
#Q14  Create a  happy new year string and then separate this string by (space) ,?
str4="happy new year"
print(str4.split())

#----------------------------------------------------------------------------------
#Q15)  Create two string and check whether one string present in another string using membership operator?
str1="happy birthday"
str2="happy"
if(str2 in str1):
    print("string 2nd is present in string 1st")
else:
    print("string 2nd is not present in string 1st")
#-----------------------------------------------------------------------------------