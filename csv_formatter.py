import pandas as pd
import numpy as np 

filename = 'pre'
data = pd.read_csv("labels/" + filename + "/" + filename + ".csv")
data = data.iloc[:, 1:33]
data.fillna(0, inplace=True)


# df2 = pd.DataFrame(columns=['roundness', 'skin_white','skin_sabze', 'skin_gandom', 'eyecolor_blue', 'eyecolor_green', 'eyecolor_tusi', 'eyecolor_asali', 'eyecolor_brown', 'eyecolor_black', 'eyesize_d', 'eyesize_r', 'eyesize_m', 'nose_flat', 'nose_fantasy', 'nose_fleshy', 'nose_skinny', 'lips_d', 'lips_m', 'lips_k', 'forhead', 'eyebrow', 'eyelash', 'cheek', 'chin'])
df = pd.DataFrame()

# face roundness
df['roundness'] = data.iloc[:, 0]
# skin color
df[['skin_white','skin_sabze', 'skin_gandom']] = data.iloc[:, 2:5]
# eye color 
df[['eyecolor_blue', 'eyecolor_green', 'eyecolor_tusi', 'eyecolor_asali', 'eyecolor_brown', 'eyecolor_black']] = data.iloc[:, 5:11]
# eye size
df[['eyesize_d', 'eyesize_r', 'eyesize_m']] = data.iloc[:, 11:14]
# nose flatness
df['nose_flat'] = data.iloc[:, 14] + data.iloc[:, 16]
# nose type
df['nose_fantasy'] = data.iloc[:, 18]
df['nose_fleshy'] = data.iloc[:, 14] + data.iloc[:, 15]
df['nose_skinny'] = data.iloc[:, 16] + data.iloc[:, 17]
# lips
df[['lips_d', 'lips_m', 'lips_k']] = data.iloc[:, 19:22] 
# forehead
df['forehead'] = data.iloc[:, 23]
# eyebrow
df['eyebrow'] = data.iloc[:, 25]
# eyelash
df['eyelash'] = data.iloc[:, 27]
# cheek 
df['cheek'] = data.iloc[:, 29]
# chin
df['chin'] = data.iloc[:,31]

# dx = df.apply(pd.to_numeric)
df = df.astype(int)
# print(df.dtypes)

df.to_csv("labels/" + filename + "/labels.csv", encoding='utf-8', sep=',', index=False)
print(df.head())
# print(df.columns)
