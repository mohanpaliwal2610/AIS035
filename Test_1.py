# Python Exercise No: 1
#Subject: Data types, Operators and Input and Output in python

#---------------------------------------------------------------
#Q1) What is variable? And rules to declare a variable in python?

'''A python variable is reserved memory location to store value.
 Variables can be re-declared even after you have declared them for once
The equal sign (=) is used to assign values to variables.

Rules:-
A variable name must start with a letter or the underscore character
A variable name cannot start with a number
A variable name can only contain alpha-numeric characters and underscores
Variable names are case-sensitive '''

#---------------------------------------------------------------
#Q2) What are the primitive built-in types in python?
'''
python has four primitive types:- integers, floats, booleans, strings
'''

#---------------------------------------------------------------
#Q3) What is the mutable and immutable object in Python, Which data types are immutable in python
'''
mutable:- mutable object can change or update after declaration
            e.g. list, mutable set, dictionary

immutable:-immutable object can not change or update after declaration
            e.g. number, string, tuple
'''
#------------------------------------------------------------
#Q4) How to get the length of the string?
'''we use 'len()' for find length of string'''
str="hello AIS Solutions"
print(str)
print(len(str))

#------------------------------------------------------------
#Q5)Which are the Relational Operators and their used in python?
'''
x==y is TRUE if x is equal to y
x!=y is TRUE if x is not equal to y
x<y  is TRUE if x is less than y
x>y  is TRUE if x is greater than y
x<=y  is TRUE if x is less than  or equal to y
x>=y  is TRUE if x is greater than or equal to y
'''
#---------------------------------------------------------------
#Q6) What is the result of float (5), 10>="9", 10**2, len ("basic python")?
print(" the result of float(5) is :-",float(5))
print(" the result of 10>=9 is :-",10>="9")
print(" the result of 10**2 is :-",10**2)
print(" the result of len(''basic python'') is :-",len ("basic python"))

#---------------------------------------------------------------
#Q7) What is meant by input and output? Which function is used to display output in python?
'''
Input means the data entered by the user of the program.
in python input() and raw_input() function available for input.

Output means display the value of python code in output window.
print() function are used in python
'''
#---------------------------------------------------------------
#Q8) Convert any integer number into float, string, and convert float into string, integer Convert string into float, integer
print("convert integer no into float :-",float(10))
print("convert integer no into string :-",str(10))
print("convert float no into string :-",str(10.10))
print("convert float no into integer :-",int(10.10))
print("convert string into integer :-",int("10"))
print("convert string into float :-",float("10"))

#---------------------------------------------------------------
#Q9)  Take integer number using input function from user?
x=int(input("enter any number :"))
print(x)

#-------------------------------------------------------------
#Q10)  Take two number using input function and find their sum?

no=2
x=list()
while(no>0):
    y=int(input("enter nomber"))
    no=no-1
    x.append(y)
print sum(x)

#------------------------------------------------------------
#Q11) Write a python program to accept two number using input function and use all the arithmetic operators?

x=int(input(" enter 1st number :"))
y=int(input(" enter 2st number :"))
print("sum is: ",x+y,"multiplication is :",x*y,"substraction x-y is :",x-y)
print("power x^y is :",x**y,"division x/y :",x/y)

#------------------------------------------------------------
#Q12) Check whether two numbers are equal or not using identity operator?

x=int(input(" enter 1st number :"))
y=int(input(" enter 2st number :"))
if(x is y):
    print("both numbers are equal")
else:
    print("both numbers are not equal")

#------------------------------------------------------------
#Q13) Write a python program to check whether two string are equal or not using comparison operator?

str1=input("inter 1st string")
str2=input("inter 2st string")
if(str1==str2):
    print("both strings are equal")
else:
    print("both strings are not equal")


#------------------------------------------------------------
#Q14). Assign same value to three variable name a, b and c using one line command?
a,b,c=10,10,10
print(a,b,c)

#--------------------------------------------------------------
#Q15) How to print address of variables in python?
x=10
print("address of x is : ",id(x))

#------------------------------------------------------------------
#Q16) Introduction of python? For eg, invention, version and installation?
'''
Python is a popular programming language.It was created in 1991 by Guido van Rossum.
Python is a high-level, interpreted, interactive and objectoriented scripting language.
Python works on different platforms (Windows, Mac, Linux, Raspberry Pi, etc)
The Python download requires about 25 Mb of disk space;
in case you need to re-install Python. When installed, Python
requires about an additional 90 Mb of disk space.
'''