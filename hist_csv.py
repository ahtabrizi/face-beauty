import pandas as pd
import numpy as np 

filename = 'post'
data = pd.read_csv("labels/" + filename + "/labels.csv")

print("Labels for " + filename.upper())
hist = {}
for c in data.columns:
   hist[c] = data[c].sum() 
   print(c, " --- ", hist[c])


