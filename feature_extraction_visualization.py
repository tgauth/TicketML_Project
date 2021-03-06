#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 19:52:15 2019

"""

from collections import Counter
import pandas as pd, numpy as np, datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

Data=pd.read_csv('Traffic_Violations.csv')

# Convert Date to day of week Monday=0
Data['Date Of Stop']=pd.to_datetime(Data['Date Of Stop']).dt.dayofweek

# Convert time to Day time(Morning, Evening,etc)
Morning=np.ndarray.tolist(pd.date_range("6:00", "12:00", freq="1min").time)
Afternoon=np.ndarray.tolist(pd.date_range("12:01", "17:00", freq="1min").time)
Evening=np.ndarray.tolist(pd.date_range("17:01", "21:00", freq="1min").time)
Night=np.ndarray.tolist(pd.date_range("21:01", "5:00", freq="1min").time)
Time=pd.to_datetime(Data['Time Of Stop']).tolist()
newTime=[]

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

data_columns = ['SubAgency', 
                'Gender', 
                'Race',
                'Date Of Stop',
                'Time Of Stop', 
                'Location', 
                'Year',
                'Latitude',
                'Longitude']

Data = Data[data_columns]
Data.to_csv("Traffic_Violations_Features.csv")

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
    plt.savefig(os.getcwd() + "/plots/{}.png".format(feature))

#featureList_vehicle = ['VehicleType', 'Year', 'Make', 'Model'] #Seems to be very inconsistent with naming for Model and Make, year and color might be our best bets here
# featureList_vehicle = ['Color','Year']


# for feature in featureList_vehicle:
#     count = Counter(Data[feature])
#     plt.pie(list(count.values()),labels = list(count.keys()))
#     plt.title(feature)
#     plt.savefig("{}.png".format(feature))