#Name: Mohan Subhash Paliwal..
#ID: AIS035
#------------------------------------------------------------------------------
#========K_means clustering (unsupervised learning) on iris datasets
# data set 1 :- iris
# -----rread data------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#=== data importing
iris=pd.read_csv("C:\\Users\\admin\\Desktop\\AIS Solutions\\iris.csv")
print(iris.head())
'''==output===
 Sepal.Length  Sepal.Width  Petal.Length  Petal.Width Species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
'''
x=iris.iloc[:,[0,1,2,3]].values

#-----------------------------------------
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",max_iter=300)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),wcss,markers='o',color='red')
plt.title("The Elbow Method")
plt.xlabel("no. of clusters")
plt.ylabel("wcss")
plt.show()

#==fitting K-means to datasets
kmeans=KMeans(n_clusters=3,init="k-means++",random_state=101)
kmeans.fit(x)
y_kmeans=kmeans.predict(x)
print(y_kmeans)
'''====output====
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 2 2 2 2 0 2 2 2 2
 2 2 0 0 2 2 2 2 0 2 0 2 0 2 2 0 0 2 2 2 2 2 0 2 2 2 2 0 2 2 2 0 2 2 2 0 2
 2 0]

'''
#visualization of clusters
plt.figure(figsize=(15,7))
sns.scatterplot(x[y_kmeans==0,0],x[y_kmeans==0,1],color="purple",label="cluster 1",s=50)
sns.scatterplot(x[y_kmeans==1,0],x[y_kmeans==1,1],color="red",label="cluster 2",s=50)
sns.scatterplot(x[y_kmeans==2,0],x[y_kmeans==2,1],color="green",label="cluster 3",s=50)
sns.scatterplot(x[y_kmeans==3,0],x[y_kmeans==3,1],color="blue",label="cluster 4",s=50)
sns.scatterplot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="grey",label="centroids",marker='^',s=100)

plt.title("clusters of flowers")
plt.legend()
plt.show()



y=iris.iloc[:,4]
print(pd.crosstab(y,y_kmeans))
"""
col_0        0   1   2
Species
setosa       0  50   0
versicolor  48   0   2
virginica   14   0  36

"""