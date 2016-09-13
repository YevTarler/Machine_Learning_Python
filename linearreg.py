
import pandas as pd
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

print(data.head())
data.tail()

# becouse the repsonse var is continous this is a regression problem

import seaborn as sns
sns.pairplot(data,x_vars=['TV','Radio','Newspaper'], y_vars='Sales',size=7,aspect=0.7, kind='reg')
# sns.plt.show()

feature_cols = ['TV','Radio','Newspaper']

#select subset of the dataset
X = data[feature_cols]
print(type(X))

y = data['Sales'] #also can be done with dot notation

#splitting X and y into training and testing sets

from sklearn.cross_validation import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#Import model and instantiate
from sklearn.linear_model import LinearRegression

linereg = LinearRegression()
linereg.fit(X_train, y_train) #fitting step different in each model, LR finds intercept and coefficients
print (linereg.intercept_)
print (linereg.coef_)
zip(feature_cols,linereg.coef_); #pait names with coefficients

#make predictions
y_pred = linereg.predict(X_test)

#RMSE - root mean squere error, most common way to evaluate. The lower the better prediction
from sklearn import metrics
import numpy as np
print (np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
