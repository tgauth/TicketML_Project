#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:35:35 2019

@author: Tess
"""

import pandas as pd, numpy as np, datetime as dt


Data=pd.read_csv('Traffic_Tickets_Issued_Four_Year_Window.csv')

#Convert Date to day of week Monday=0
Data['Date Of Stop']=pd.to_datetime(Data['Date Of Stop']).dt.dayofweek

#Convert time to Day time(Morning, Evening,etc)

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
    
