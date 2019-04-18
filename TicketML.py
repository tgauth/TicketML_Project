#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:35:35 2019

@author: Tess
"""

import pandas as pd, numpy as np, datetime as dt
import random
 
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