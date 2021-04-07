# GRIP--Task-2
#Importing all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import datasets

#Importing dataset
df1 = datasets.load_iris()

df = pd.DataFrame( df1.data, columns= df1.feature_names)

df

#Displays all the unique values for each column in the given datasheet
df.nunique()

#Giives the correlation matrix
df.corr()

df['sepal length (cm)'].unique()

#Plotting of sepal length and petal length
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c= 'red')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
#plot of comparison between sepal length and petal length
plt.title('Sepal vs Petal length')
plt.show()

#Plotting of sepal width and petal width
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c= 'blue')
plt.xlabel('Sepal width')
plt.ylabel('Petal width')
#plot of comparison between sepal width and petal width
plt.title('Sepal vs Petal width')
plt.show()

#plotting the histogram
#histogram displays numerical data by grouping data into "bins" of equal width.
df['sepal length (cm)'].hist(bins=25)

df['sepal width (cm)'].hist(bins=25)

df['petal length (cm)'].hist(bins=25)

df['petal width (cm)'].hist(bins=25)

#this function enables us to select a particular cell of the datasheet
x = df.iloc[:, :].values

#Elbow method
from sklearn.cluster import KMeans
WCSS=[]

for i in range(1,12):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(x)
    WCSS.append(kmeans.inertia_)
plt.plot( range(1,12), WCSS, C='magenta')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Optimum number of cluster by the Elbow method')
plt.show()

kmeans= KMeans( n_clusters= 3, random_state=42)
predicted_y= kmeans.fit_predict(x)

#visualizing the cluster
plt.scatter(x[predicted_y==0 ,0], x[predicted_y==0, 1], s=150, c='Aqua', label='Iris-setosa')
plt.scatter(x[predicted_y==1 ,0], x[predicted_y==1, 1], s=150, c='Magenta', label='Iris-versicolor')
plt.scatter(x[predicted_y==2 ,0], x[predicted_y==2, 1], s=150, c='purple', label='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 102, c = 'yellow', label = 'Centroids')
plt.xlabel('abs')
plt.ylabel('idk')
plt.legend()
