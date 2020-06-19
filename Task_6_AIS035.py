#Name: Mohan Subhash Paliwal..
#ID: AIS035


#Python Exercise No: 6
#Subject: Control Statements
#-----------------------------------------------------
#Q1)Which are the control statements in python?
'''
loop control statement : 1) break statement 2) continue statement 3) pass statement
'''
#-----------------------------------------------------
#Q2 A python program to test whether a given number is in between 1 and 10?

import numpy as np
import pandas as pd
import scipy as sp
no=sp.random.randint(1,30,5)
for x in no:
    if (x in range(1,11)):
         pass
         print(x)

#-----------------------------------------------------
#q3)Write a python program to check whether number is even or odd?
x=int(input("inter any number: "))
if(x%2==0):
    pass
    print(" given no is even")

#-----------------------------------------------------
#Q4) Write a python program to check whether a list is empty or not using len () function?
ls=[1,2,3]
n=len(ls)
if(n==0):
    print("the list is empty")
else:
    print("the list is not empty , size of list is ",n)
#-----------------------------------------------------
#Q5)A python program to check whether a given number is zero, positive or negative?
a=int(input("inter any number: >"))
if(a==0):
    print("no is zero")
elif(a>0):
    print("no is positive")
else:
    print(" no is negative")

#-----------------------------------------------------
#Q6)Print first 10 natural numbers using while loop and for loop?
i=1
while(i<=10):
    print(i)
    i=i+1

for x in range(1,11):
    print(x)
#-----------------------------------------------------
#Q7)Write a python program to reverse the following list using for loop?
# lt1 = [12, 7, 9, 14, 5]
ls1=[12,7,9,14,5]
new=[]
for x in range(len(ls1)):
    new.insert(i,ls1[-1])
    ls1.pop(-1)
print(new)
#-----------------------------------------------------
#Q8)Write a python program to multiply all the items in a list using for loop?
ls1=[12,7,9,14,5]
new=1
for x in (ls1):
    new=new*x
print(new)

#-----------------------------------------------------
#Q9)Write a python program to print numbers in list which are divisible by 6 and if number is
#greater than 150 then stop the iteration?
#list1 = [12, 15, 36, 20, 55, 47, 102, 150, 146, 180]
list1=[12,15,36,20,55,47,102,150,146,180]
for x in list1:
    if(x>=150):
        break
    elif(x%6==0):
        print(x)

#------------------------------------------------------
#Q10) Write a program to count a specific letter in the string.?
str1="hello ais solution,how are you"
x=str(input("inter any specific letter : >"))
no=0
for i in str1:
    if(x==i):
        no=no+1
print(no)
#-----------------------------------------------------------------------------------
