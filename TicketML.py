#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:35:35 2019

@author: Tess
"""

from collections import Counter
import pandas as pd, numpy as np, datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


Data=pd.read_csv('Traffic_Violations.csv')

# Convert Date to day of week Monday=0
Data['Date Of Stop']=pd.to_datetime(Data['Date Of Stop']).dt.dayofweek

# Convert time to Day time(Morning, Evening,etc)
Morning=np.ndarray.tolist(pd.date_range("6:00", "12:00", freq="1min").time)
Afternoon=np.ndarray.tolist(pd.date_range("12:01", "17:00", freq="1min").time)
Evening=np.ndarray.tolist(pd.date_range("17:01", "21:00", freq="1min").time)
Night=np.ndarray.tolist(pd.date_range("21:01", "5:00", freq="1min").time)
Time=Data['Time Of Stop'].tolist()
newTime=[]
for t in Time:
    if t in Morning:
        newTime.append('Morning')
    elif t in Afternoon:
        newTime.append('Afternoon')
    elif t in Evening:
        newTime.append('Evening')
    else:
        newTime.append('Night')
        
Data['Time Of Stop']=newTime

# Feature Visualization 
# intended for features with a small number of options (i.e. yes or no)

# show disparity among options
featureList = ['SubAgency', 'Gender', 'Race']

# don't graph well as bar plot
# featureList = ['Driver City', 'Driver State', 
#               'DL State', 'Arrest Type', 'Description', 'Fatal']

# only 1 option, or options end up being basically even
# featureList = ['Agency', 'Contributed To Accident', 'Belts', 'Accident'
#               'Personal Injury', 'Property Damage', 'Commercial License', 
#               'HAZMAT', 'Commercial Vehicle', 'Work Zone', 'State']

# not graphed but possibly important
# featureList = ['Location', 'Latitude', 'Longitude', 'Geolocation']
# featureList = ['Vehicle Type', 'Year', 'Make', 'Model']


for feature in featureList:
    count = Counter(Data[feature])
    plt.bar(range(len(count)), list(count.values()), align='center')
    plt.xticks(range(len(count)), list(count.keys()))
    plt.title(feature)
    plt.savefig("{}.png".format(feature))