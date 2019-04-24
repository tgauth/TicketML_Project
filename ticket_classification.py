# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:04:21 2019
 
@author: Tess
"""
 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# to be read in after negative sampling is complete
df = pd.read_csv("Traffic_Violations_withNegs.txt", sep='\t', engine='python')
for col in df.columns: 
    print(col)  
feature_columns = ['SubAgency', 'Gender', 'Race','Time Of Stop']
test_df = df[feature_columns]
 
# https://towardsdatascience.com/predicting-vehicle-accidents-with-machine-
# learning-ce956467fa74
# lblE = LabelEncoder()
# for i in test_df:
#     if test_df[i].dtype == 'object':
#         lblE.fit(test_df[i])
#         test_df[:, i] = lblE.transform(test_df[i])
       
## Let us split our data into training and validation sets
#X_train, X_test, y_train, y_test = train_test_split(df.drop('fatalCount', axis=1), df.fatalCount, test_size=0.33, random_state=42)
# Need to drop "Ticket" column from dataframe and use as "Y"
 
""" Random Forest Regressor """
#m = RandomForestRegressor(n_estimators=50)
#m.fit(X_train, y_train)
#print_score(m)
 
""" Multinomial NB """
#clf = MultinomialNB()
#clf.fit(X_train, y_train)
#clf.predict(X_test)