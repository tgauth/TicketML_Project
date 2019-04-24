#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:35:35 2019

@author: Tess
"""

import pandas as pd

Data=pd.read_csv('Traffic_Violations_Features.csv')
    
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
Data.to_csv("Traffic_Violations_With_Negatives.csv", sep='\t')
