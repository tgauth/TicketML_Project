#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:35:35 2019

@author: Tess
"""

from collections import Counter
import pandas as pd, numpy as np, datetime as dt
<<<<<<< HEAD:build_neg_data.py
import random
 
Data=pd.read_csv('Traffic_Tickets_Issued_Four_Year_Window.csv')
=======
import matplotlib.pyplot as plt
import seaborn as sns
import math


Data=pd.read_csv('Traffic_Violations.csv')
>>>>>>> master:TicketML.py

# Convert Date to day of week Monday=0
Data['Date Of Stop']=pd.to_datetime(Data['Date Of Stop']).dt.dayofweek

# Convert time to Day time(Morning, Evening,etc)
Morning=np.ndarray.tolist(pd.date_range("6:00", "12:00", freq="1min").time)
Afternoon=np.ndarray.tolist(pd.date_range("12:01", "17:00", freq="1min").time)
Evening=np.ndarray.tolist(pd.date_range("17:01", "21:00", freq="1min").time)
Night=np.ndarray.tolist(pd.date_range("21:01", "5:00", freq="1min").time)
Time=pd.to_datetime(Data['Time Of Stop']).tolist()
newTime=[]
#for t in Time:
#    if t in Morning:
#        newTime.append('Morning')
#    elif t in Afternoon:
#        newTime.append('Afternoon')
#    elif t in Evening:
#        newTime.append('Evening')
#    else:
#        newTime.append('Night')
for t in Time:
    hour = t.hour
    if hour >= 6 and hour <= 12:
        newTime.append('Morning')
    elif hour > 12 and hour <= 17:
        newTime.append('Afternoon')
    elif hour > 17 and hour <= 21:
        newTime.append('Evening')
    elif hour > 21 or hour < 6:
        newTime.append('Night')
Data['Time Of Stop']=newTime
<<<<<<< HEAD:build_neg_data.py
    
# Assign all current data as a positive sample

numRows = Data.shape[0]
Labels= [1] * numRows

# Flesh out negative samples

# Randomly Select a row and a column
for i in range(numRows * 3):
    randRow1 = random.randint(0, numRows - 1)
    randRow2 = random.randint(0, numRows - 1)
    randCol = random.randint(1, Data.shape[1] - 1) # First column seems to be an index, we don't want that
    newSample = Data.iloc[randRow1,:]
    newSample[randCol] = Data.iloc[randRow2,randCol]
    Data.loc[Data.shape[0]] = newSample
    Data.duplicated(keep='first')
    if(Data.shape[0] != numRows):
        Labels.append(0)
        numRows = numRows + 1
    else:
        print("There is a repeat")
Data['Label'] = Labels
Data.to_csv("Traffic_Tickets_Issued_Four_Year_Window_With_Negatives.csv", sep='\t')
=======

Year=(Data['Year']).tolist()
newyear=[]

for y in Year:
    if math.isnan(y):
        newyear.append('None')
    elif y > 2019 or y < 1900:
        newyear.append('None')
    elif y > 2015:
        newyear.append('New')
    elif y > 2005 and y < 2015:
        newyear.append('Modern')
    elif y > 1980 and y < 2005:
        newyear.append('Old')
    elif y < 1980:
        newyear.append('Classic')
    else:
        newyear.append('None')


Data['Year']=newyear

# Feature Visualization 
# intended for features with a small number of options (i.e. yes or no)

# show disparity among options
featureList = ['SubAgency', 'Gender', 'Race', 'Time Of Stop','Color','Year']

# don't graph well as bar plot
# featureList = ['Driver City', 'Driver State', 
#               'DL State', 'Arrest Type', 'Description', 'Fatal']

# only 1 option, or options end up being basically even
# featureList = ['Agency', 'Contributed To Accident', 'Belts', 'Accident'
#               'Personal Injury', 'Property Damage', 'Commercial License', 
#               'HAZMAT', 'Commercial Vehicle', 'Work Zone', 'State']

# not graphed but possibly important
# featureList = ['Location', 'Latitude', 'Longitude', 'Geolocation']

for feature in featureList:
    count = Counter(Data[feature])
    print(count)
    plt.bar(range(len(count)), list(count.values()), align='center')
    plt.xticks(range(len(count)), list(count.keys()))
    plt.title(feature)
    plt.savefig("{}.png".format(feature))



#featureList_vehicle = ['VehicleType', 'Year', 'Make', 'Model'] #Seems to be very inconsistent with naming for Model and Make, year and color might be our best bets here
# featureList_vehicle = ['Color','Year']


# for feature in featureList_vehicle:
#     count = Counter(Data[feature])
#     plt.pie(list(count.values()),labels = list(count.keys()))
#     plt.title(feature)
#     plt.savefig("{}.png".format(feature))
>>>>>>> master:TicketML.py