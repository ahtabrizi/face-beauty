import numpy as np 
import pandas as pd 


quality_pre = pd.read_csv("./results/Qualitative_pre.csv",header=None)
quality_post = pd.read_csv("./results/Qualitative_post.csv",header=None)
quantity_pre = pd.read_csv("./results/Quantitative_pre.csv",header=None)
quantity_post = pd.read_csv("./results/Quantitative_post.csv",header=None)

df = pd.DataFrame(columns=["quality_pre", "quality_post", "quantity_pre", "quantity_post", "score_pre", "score_post", "improvement", "improvement_percent"])

df["quality_pre"] = quality_pre.sum(axis=1)
df["quality_post"] = quality_post.sum(axis=1)
df["quantity_pre"] = quantity_pre.sum(axis=1)
df["quantity_post"] = quantity_post.sum(axis=1)

df["score_pre"] = df["quality_pre"] + df["quantity_pre"] 
df["score_post"] = df["quality_post"] + df["quantity_post"] 

df["improvement"] = df["score_post"] - df["score_pre"]
df["improvement_percent"] = 100 * df["improvement"] / (df["score_pre"] + 0.01)

df.to_csv("results/final_scores.csv", encoding='utf-8', sep=',', index=False)
