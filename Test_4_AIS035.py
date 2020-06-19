#Name: Mohan Subhash Paliwal..
#ID: AIS035

#Python Exercise No: 4
#Subject: Dictionary and Set in python

#----------------------------------------------------------------
#Q1) What is meant by dictionary?
'''
Dictionary is key Value pair data format.
it is mutable,unordered sequence indexces does not start from 0.
key must be immutable and unique.
declaired using {} bracket.
key and values seperated by colon.
'''
#---------------------------------------------------------------
#Q2) What is difference between dictionary and set?
'''
dictionary is key-value data farmat and set is single format.
dictionaries are unorederd sets. but the main difference is that items in dictionaries are
accessed via keys and not via their position.the value of a dictionary can be any python datatype.
A set is a collection which is unorederd and unindexed. so dictionaries are unorderd key value pairs.
'''
#--------------------------------------------------------------
#Q3) Create a python set of mixed data types?
s1={1,"HELLO",2.45,(5,4,3)}
print(s1)
#-------------------------------------------------------------
#Q4) Create a Dictionary of employee name, Id and salary?

d1={"name":["sam","rohit"],"ID":[2022,2023],"salary":[50000,60000]}
print(type(d1))
print(d1)

#-----------------------------------------------------------
#Q5) Display key-value pairs from dictionary using items () method?
d1={"name":["sam","rohit"],"ID":[2022,2023],"salary":[50000,60000]}
print(d1.items())
#------------------------------------------------------------
#Q6)  Create a dictionary and access values of 2nd key from the dictionary?
d1={"name":["sam","rohit"],"ID":[2022,2023],"salary":[50000,60000]}
print(d1["ID"])

#------------------------------------------------------------
#Q7) Create a dictionary and then print empty dictionary using clear () method?
d1={"name":["sam","rohit"],"ID":[2022,2023],"salary":[50000,60000]}
print(d1.clear())
#-----------------------------------------------------------
#Q8) Create a Dictionary of name, roll no, marks, gender and access keys from the dictionary using key () method?
d2={"name": ["mohan","supri","sunil","niku"],
    "roll no":[32,43,40,12],
    "marks":[80,99,75,86],
    "gender":["male","female","male","female"]}
print(d2)
print(d2.keys())
#-----------------------------------------------------------
#Q9) Take set A for even and B for odd and print the union of two set?
A={2,4,6,8,10,12}
B={1,3,5,7,9,11}
print(A|B)
#--------------------------------------------------------------
#Q10) If A={1,2,3,4,5} and B={4,5,6,7,8} then find difference, intersection of set A and B?
A={1,2,3,4,5}
B={4,5,6,7,8}
print("difference of A-B is :",A-B)
print("difference of B-A is :",B-A)
print("intersection of A and B is :",A&B)

#--------------------------------------------------------------
#Q11) If set= {1, 2, 3, 5, 6} then add an element 4 in the set using add () function and remove 3 using discard () function?
set={10,2,3,5,6}
set.add(4)
print(set)
set.discard(3)
print(set)
#------------------------------------------------------------
#Q12) Add 12,9,11 in the set A = {1, 2, 3, 5, 6} using update () function?
A={1,2,3,5,6}
A.update({12,9,11})
print(A)
#----------------------------------------------------------
#Q13) Create a set and print the sum of all the elements in the set?
s3={11,2,14,50,40,30,23,75,5,18}
print("the sum of all element of",s3,"is ",sum(s3))
#--------------------------------------------------------
#Q14) if myset={"a","p","p","l","e"} then check whether a and r is in the set or not using
#membership operator?
myset={"a","p","p","l","e"}
print(myset)
s4={"r","a"}

if(s4&myset==s4):
    print("a and r are member of myset")
else:
    print("a and r are not member of myset")


print({"a","r"} not in myset)
#--------------------------------------------------------------
#Q15)  Create a dictionary of players name and runs in match and print keys and values in
#dictionary using for loop?

players={"name":["dhoni","sehvag","tendulkar","virat"],
         "runs":[54,101,86,75]}
print(players)
key=players.keys()
print(key)
val=players.values()
print(val)
for i in key:
    print(i)

for i in val:
    print(i)
#------------------------------------------------------------
