import matplotlib
import numpy
import sklearn

from sklearn.datasets import load_iris
#model for classification
from sklearn.neighbors import KNeighborsClassifier
#model 2 for classification, as learned in coursera
from sklearn.linear_model import LogisticRegression

iris = load_iris()

X = iris.data
y = iris.target

print(X.shape)
print(y.shape)

#terminology
#each row - observation / record
#each column - feature/predictor/attribute
#what we are gonna predict(y) - target/outcome/response

# seperates the data as a map of colors, and classifies areas:
knn = KNeighborsClassifier(n_neighbors=1)
print (knn)
#fit the model with data: model learn the relationship
knn.fit(X,y)

X_new = [[3,5,4,2],[5,4,3,2]]
print(knn.predict(X_new)) #works as numpy array or python array

#logistic regression
logreg = LogisticRegression()
logreg.fit(X,y)
print("logistic regression: ")
y_pred = logreg.predict(X)

from sklearn import metrics
print (metrics.accuracy_score(y, y_pred)) # 0.96 output. so 96% is correct

y_pred_knn = knn.predict(X)
print (metrics.accuracy_score(y, y_pred_knn))

# test acurecy with diffrent K's
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4) #test_size 40% will be testing set. best betweeen 20-40%

k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

# import Matplotlib (schientific plotting library)
import matplotlib.pyplot as plt


plt.plot(k_range,scores)
plt.xlabel('value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()