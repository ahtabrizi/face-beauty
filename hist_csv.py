import pandas as pd
import numpy as np 

filename = 'post'
data = pd.read_csv("labels/" + filename + "/labels.csv")

print("Labels for " + filename.upper())
hist = {}
for i, c in enumerate(data.columns):
   hist[c] = data[c].sum() 
   print(i + 1, c, " --- ", hist[c])


