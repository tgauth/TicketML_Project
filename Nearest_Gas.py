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

Data=pd.read_csv('Traffic_Violations_With_Negatives_48544.csv',error_bad_lines=False)