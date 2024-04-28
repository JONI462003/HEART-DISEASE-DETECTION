1) Data Pre-processing

import pandas as pd
import numpy as np

df=pd.read_csv(r"/content/cardio_train - cardio_train.csv.csv")
df.head(200)

2) Data Analysis and Visualization

import matplotlib.pyplot as plt
import seaborn as sns
# Histogram of age
plt.figure(figsize=(8, 6))
sns.histplot(df['age'], kde=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()

#barplot of sex
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='gender', hue='active')
plt.xlabel('Sex (0:Female, 1:Male)')
plt.ylabel('Count')
plt.title('Distribution of Sex')
plt.show()

3) Correlation Matrix

correlation_matrix = df.corr()

# Plot the correlation matrix using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

4) Machine Learning Techniques

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x=df.drop("cardio",axis="columns")
y=df["cardio"]

x

y

(A) SVM - Support Vector Machine

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)

from sklearn.svm import SVC
model=SVC()

model.fit(xtrain,ytrain)

pred=model.predict(xtest)
pred

model.score(xtest,ytest)

(B) KNN - K-Nearest Neighbour

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

knn.fit(xtrain,ytrain)

knn.predict(xtest)

knn.score(xtest,ytest)

(C) DT - Decision Tree

import pandas as pd
import matplotlib.pyplot as plt

inputs=df.drop("cardio",axis="columns")
target=df["cardio"]

inputs

target

(D) LR - Logistic Regression

 Here, train_test_split will be used

train_test_split:
The data can be divided or splitted into two categories. One for Training Data and another one for Testing data.


To build a machine learning model, we need to
(i)train the model  { 80% }
(ii)test the model  { 20% }


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)

len(xtrain)

len(xtest)

xtrain

xtest

from sklearn.linear_model import LogisticRegression
logireg=LogisticRegression()

logireg.fit(xtrain,ytrain)

Predicted=logireg.predict(xtest)

Predicted

logireg.score(xtest,ytest)

(E) RF - Random Forest

It is a Supervised Classification Algorithm
           It consists of so many decision trees.

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=50)

model2=RandomForestClassifier(n_estimators=50,criterion="gini")
model2.fit(xtrain,ytrain)

model2.score(xtest,ytest)

5) Machine Learning Model for Heart Disease Detection


(A) SVM - Support Vector Machine
      0.5967380952380953

(B) KNN - K-Nearest Neighbour
      0.5565714285714286

(C) DT - Decision Tree
      It segregates the data

(D) LR - Logistic Regression
      0.6929285714285714

(E) RF - Random Forest
      0.7199285714285715

So, from these accuracy points we can consider Random Forest Classifier as the best suited model to solve the Heart Disease Detection Query.
