#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ahmernajar

"""

#Titanic
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train (1).csv')
X = dataset.iloc[:, [2,4,5,6,7,9,11]].values
y = dataset.iloc[:, 1].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', 
                                    random_state = 0)
classifier.fit(X_train, y_train)

#Fitting DecisionTree to the Training ser
from sklearn.tree import DecisionTreeClassifier
classifierr = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierr.fit(X_train, y_train)
y_pred = classifierr.predict(X_test)


# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('test (1).csv')
test_dataset = dataset1.iloc[:, [1,3,4,5,6,8,10]].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(test_dataset[:, :])
test_dataset[:, :] = imputer.transform(test_dataset[:, :])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_a = LabelEncoder()
test_dataset[:, 1] = labelencoder_test_dataset.fit_transform(test_dataset[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [0])
test_dataset = onehotencoder.fit_transform(test_dataset).toarray()

test_dataset[:, 6] = labelencoder_a.fit_transform(test_dataset[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [0])
test_dataset = onehotencoder.fit_transform(test_dataset).toarray()


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test_dataset = sc.fit_transform(test_dataset)


# Predicting the Test set results
final_pred = classifier.predict(test_dataset)

