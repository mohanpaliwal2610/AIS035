#Name: Mohan Subhash Paliwal..
#ID: AIS035

#Python Exercise No: 5
#Subject: Basic programs
#--------------------------------------------------
#Q1) Create a tuple to display the values of a tuple using for loop?
x=(10,25,55,48,66,85,44)
len(x)
for i in x:
    print(i)

#--------------------------------------------------
#Q2)Write a python program to display even numbers between 100 and 120 using while loop?

x=100
while x<121:
    if(x%2==0):
        print(x)
    x=x+1

#--------------------------------------------------
#Q3)  Write a python program to display numbers from 10 to 1 in decreasing order using for loop?
x=range(1,11)
for i in range(1,11):
    print(x[-i])

#--------------------------------------------------
#Q4) Write a python program to display numbers from 11 to 15 using continue statement?

x=range(1,100)
for i in x:
    if((i<11)|(i>15)):
        continue
    print(i)

#--------------------------------------------------
#Q5) Write a function add () to find sum of two numbers?
def add():
    a=int(input("inter 1st no :>>"))
    b=int(input("inter 2st no :>>"))
    c=a+b
    print("addition is : >>",c)

add()
#--------------------------------------------------
#Q6) Write a function multi () to find multiplication of two numbers using return statement?

def multi():
    a = int(input("inter 1st no :>>"))
    b = int(input("inter 2st no :>>"))
    c = a*b
    return(c)
multi()
#--------------------------------------------------
#Q7) Write a python program to print numbers in list which are greater than 50?
# Where, list=[12,55,9,30,47,56,23,60,75]
list=[12,55,9,30,47,56,23,60,75]
i=0
while(i<len(list)):
    if(list[i-1]>50):
        print(list[i-1])
    i=i+1

#--------------------------------------------------
#Q8) Write a function max () that returns a maximum of two number?
def max():
    a = int(input("inter 1st no :>>"))
    b = int(input("inter 2st no :>>"))
    if(a>b):
        print(a)
    else:
        print(b)
max()
#--------------------------------------------------
#Q9)  Write a function divisible () that takes number
#Q9.1)  If number is divisible by 3, it should return "yes
#Q9.2) . If number is divisible by 5, it should return "no"
#9.3) If number is divisible by both 3 and 5, it should return "yesno"
#Q9.4)  Otherwise it should return "notdivisible"

def divisible():
    a = int(input("inter 1st no :>>"))
    if((a%3==0)&(a%5==0)):
        print("yesno")
    elif(a%5==0):
        print("no")
    elif(a%3==0):
        print("yes")
    else:
        print("notdivisible")
divisible()
#--------------------------------------------------
#Q10) Write a function called stars (rows). If rows is 4, it should print the following?

def stars():
    a= int(input("inter row no :>>"))
    i=1
    while(i<=a):
        print("*"*i)
        i=i+1
stars()
#-------------------------------------------------


