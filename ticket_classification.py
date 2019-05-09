# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:04:21 2019

"""
 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# to be read in after negative sampling is complete
test_df = pd.read_csv("Traffic_Violations_With_5_Times_Negatives.csv", sep='\t', engine='python')
# for col in df.columns: 
#     print(col)  
    
# below is only if needed to further restrict features used 
feature_columns = ['SubAgency', 'Gender', 'Race','Date Of Stop','Time Of Stop','Location','Year']
 
# https://towardsdatascience.com/predicting-vehicle-accidents-with-machine-
# learning-ce956467fa74
lblE = LabelEncoder()
for i in test_df:
    if test_df[i].dtype == 'object':
        lblE.fit(test_df[i])
        test_df[i] = lblE.transform(test_df[i])

x = test_df[feature_columns]
y = test_df[['Label']]

## Let us split our data into training and validation sets
#X_train, X_test, y_train, y_test = train_test_split(df.drop('fatalCount', axis=1), df.fatalCount, test_size=0.33, random_state=42)
# Need to drop "Ticket" column from dataframe and use as "Y"
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42) 

""" Random Forest Regressor """
m = RandomForestRegressor(n_estimators=50)
m.fit(X_train, y_train.values.ravel())
print(m.feature_importances_)
Y_pred = m.predict(X_test)
mse = mean_squared_error(y_test, Y_pred)
print(mse)
#print_score(m)
 
""" Multinomial NB """
clf = MultinomialNB()
clf.fit(X_train, y_train.values.ravel())
Y_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, Y_pred)
print(mse)

# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
""" PCA """
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train_pca = scaler.transform(X_train)
X_test_pca = scaler.transform(X_test)
pca = PCA(n_components=4)
pca.fit(X_train_pca)
X_train_pca = pca.transform(X_train_pca)
X_test_pca = pca.transform(X_test_pca)

""" Logistic Regression """
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train_pca, y_train.values.ravel())
y_pred_pca = logisticRegr.predict(X_test_pca)
mse = mean_squared_error(y_test, y_pred_pca)
print(mse)

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
""" Decision Tree """
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train_pca, y_train)
y_pred_pca = clf.predict(X_test_pca)
score = clf.score(X_test_pca, y_test)
mse = mean_squared_error(y_test, y_pred_pca)
print(mse)