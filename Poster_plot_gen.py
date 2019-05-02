


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
import numpy as np
import matplotlib.pyplot as plt

# to be read in after negative sampling is complete
test_df = pd.read_csv("Traffic_Violations_With_5_Times_Negatives.csv", sep='\t', engine='python')
# for col in df.columns: 
#     print(col)  
T=test_df.copy()    
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





X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42) 



Ntrain=X_train.shape[0]
Ntest=X_test.shape[0]




Error=[]
Algorithm=[]





""" Random Forest Regressor """
Algo=' Random Forest Regressor'
m = RandomForestRegressor(n_estimators=50)
m.fit(X_train, y_train.values.ravel())
print(m.feature_importances_)
Y_pred = m.predict(X_test)
mse = mean_squared_error(y_test, Y_pred)
Error.append(mse)
Algorithm.append(Algo)
print(mse)
importance=m.feature_importances_
importance=np.ndarray.tolist(importance)
features=X_train.columns.tolist()
plt.bar(features, importance, align="center")
plt.title("%s Feature Importance  " % (Algo))
plt.xticks(rotation=90)
plt.show()




""" Multinomial NB """
Algo='Multinomial NB'
clf = MultinomialNB()
clf.fit(X_train, y_train.values.ravel())
Y_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, Y_pred)
Error.append(mse)
Algorithm.append(Algo)
print(mse)




""" Logistic Regression """
Algo='Logistic Regression'
scaler = StandardScaler()
scaler.fit(X_train)
X_train_Array = scaler.transform(X_train)
X_test_Array= scaler.transform(X_test)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train_Array, y_train.values.ravel())
y_pred = logisticRegr.predict(X_test_Array)
mse = mean_squared_error(y_test, y_pred)
Error.append(mse)
Algorithm.append(Algo)
print(mse)





""" Decision Tree """
Algo='Decision Tree'
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train_Array, y_train)
y_pred = clf.predict(X_test_Array)
score = clf.score(X_test_Array, y_test)
mse = mean_squared_error(y_test, y_pred)
Error.append(mse)
Algorithm.append(Algo)
print(mse)
importance=clf.feature_importances_
importance=np.ndarray.tolist(importance)
features=X_train.columns.tolist()
plt.bar(features, importance, align="center")
plt.title("%s Feature Importance  " % (Algo))
plt.xticks(rotation=90)
plt.show()




plt.plot(Algorithm,Error)
plt.title("Risk Comparison between Classifiers with Ntrain=%i and Ntest= %i " % (Ntrain,Ntest))
plt.xticks(rotation=90)
plt.show()





