#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:35:35 2019

"""

import pandas as pd, numpy as np,random
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

def Nearest_Gas_Station(Data,Gas_Stations):
    

    lat=Data['Latitude'].tolist()
    long=Data['Longitude'].tolist()
    new_Location=[]
    for k in range(0,len(lat)):
        street = (lat[k], long[k])
        store_dist=[]
        for j in Gas_Stations:
            
            gas_st=(j.latitude,j.longitude)
            store_dist.append(geodesic(street, gas_st).miles)
        shortest=min(store_dist)
        i=store_dist.index(shortest)
        new_Location.append(Gas_Stations[i].address)
    
    Data['Location']=new_Location
    return Data
        
Data=pd.read_csv('Traffic_Violations_Features.csv')
Data=Data.loc[pd.isnull(Data['Latitude'])==False]   ##Select only data that has coordinates
Data=Data.reset_index(drop=True)
#Data=Data.iloc[0:100,0:len(Data.columns)]

# Assign all current data as a positive sample

numRows = Data.shape[0]
Labels= [1] * numRows
#n_onelabels=len(Labels)   #Balance out labels
#n_zerolabels=0

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
        #n_zerolabels+=1
       
        numRows = numRows + 1
    else:
        print("There is a repeat")
    
     #if  n_zerolabels==n_onelabels:   Balance out labels
       # break
Data['Label'] = Labels

geolocator = Nominatim(user_agent="Enter your email")
Gas_Stations = geolocator.geocode("Gas Stations in Montgomery County Maryland ",exactly_one=False,timeout=10,limit=100)
Data=Nearest_Gas_Station(Data,Gas_Stations)

feature_columns = ['SubAgency', 
                   'Gender', 
                   'Race',
                   'Date Of Stop'
                   'Time Of Stop', 
                   'Location', 
                   'Year']

classification_df = Data[feature_columns]

classification_df.to_csv("Traffic_Violations_With_Negatives.csv", sep='\t')
