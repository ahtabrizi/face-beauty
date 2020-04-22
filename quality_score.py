import numpy as np 
import pandas as pd 

        
def processRow(row):
    roundness = True if row[0] > 0.5 else False # True == round  
    skin_color = (row[1:4] / np.max(row[1:4])) == 1
    eye_color = (row[4:7] / np.max(row[4:7])) == 1
    eye_size = (row[7:10] / np.max(row[7:10])) == 1
    nose_flat = True if row[10] > 0.5 else False # True == Flat
    nose_type = (row[11:14] / np.max(row[11:14])) == 1
    lips = (row[14:17] / np.max(row[14:17])) == 1
    forehead = True if row[17] > 0.5 else False # True == Normal
    eyebrow = True if row[18] > 0.5 else False # True == Normal
    eyelash = True if row[19] > 0.5 else False # True == Normal
    cheek = True if row[20] > 0.5 else False # True == Normal
    chin  = True if row[21] > 0.5 else False # False  == Normal

    # calculate table look up stuff   
    j = np.where(skin_color == True)[0][0] + 0 if roundness else 3 
    eye_color_i = np.where(eye_color == True)[0][0]
    eye_size_i = np.where(eye_size == True)[0][0] + 3

    tmp = np.where(nose_type == True)[0][0]
    if tmp == 0: # fantasy
        nose_type_i = 4 + 6
    else:
        nose_type_i = (tmp - 1) * 2 + 0 if nose_flat else 1 + 6

    lips_i = np.where(lips == True)[0][0] + 11

    table_indices = [[eye_color_i, nose_type_i, eye_size_i, lips_i], j+1]
    nontable_results = [forehead, eyebrow, eyelash, cheek, chin]
    return table_indices, nontable_results

def scoreTable(indices, tables, isMale=False):
    i, j = indices
    if isMale:
        return (0.6 * np.asarray(tables['W2M'].iloc[i,j]) + 0.4 * np.asarray(tables['M2M'].iloc[i,j])) 
    else:                                              
        return (0.6 * np.asarray(tables['M2W'].iloc[i,j]) + 0.4 * np.asarray(tables['W2W'].iloc[i,j])) 
        
def scoreNonTable(NT):
    score = np.zeros(len(NT))

    score[0] = 7.5 if NT[0] else 10 # forehead
    score[1] = 7.5 if NT[1] else 10 # eyebrow
    score[2] = 7.5 if NT[2] else 10 # eyelash
    score[3] = 7.5 if NT[3] else 10 # cheek
    score[4] = 5 if NT[4] else 10   # chin

    return score
    
def score(row, tables, isMale=False):
    table_indices, nontable_results = processRow(row)
    table_scores = scoreTable(table_indices, tables, isMale)
    nontable_scores = scoreNonTable(nontable_results)
    
    scores = np.zeros(9)
    scores[0] = nontable_scores[0]
    scores[1] = nontable_scores[1]
    scores[4] = nontable_scores[2]
    scores[6] = nontable_scores[3]
    scores[8] = nontable_scores[4]

    scores[2] = table_scores[0] 
    scores[3] = table_scores[1] 
    scores[5] = table_scores[2] 
    scores[7] = table_scores[3] 
    
    return scores 

males = [4, 12, 13, 14, 17, 31, 47, 48, 56, 57, 61, 71, 72, 77, 92, 101,
103, 109, 111, 121, 133, 153, 161, 163, 164, 167, 170, 179, 190, 198,
201, 205, 222, 229, 239, 251, 252, 255, 271, 272, 285, 310, 318, 347, 354, 369]

# first 376 is post, rest is pre
data = np.load("./finalModel_resnet_finetune.npy")
tables = {}
tables['M2W'] = pd.read_csv("./labels/M2W.csv") 
tables['M2M'] = pd.read_csv("./labels/M2M.csv") 
tables['W2W'] = pd.read_csv("./labels/W2W.csv") 
tables['W2M'] = pd.read_csv("./labels/W2M.csv") 

w, h = data.shape

TAG_NAME = "post"
df = pd.DataFrame(columns=["forehead", "eyebrow", "eyecolor", "nose", "eyelash", "eyesize", "cheek", "lips", "chin"])
for i in range(w):
    if i == 376:
        TAG_NAME = "pre"
        df.to_csv(SAVE_NAME, encoding='utf-8', sep=',', index=False, header=False)
        del df
        df = pd.DataFrame(columns=["forehead", "eyebrow", "eyecolor", "nose", "eyelash", "eyesize", "cheek", "lips", "chin"])
    SAVE_NAME = "results/Qualitative_" + TAG_NAME + ".csv"


    row = data[i, :]
    k = i
    if TAG_NAME == "pre":
        k = k - 376
    isMale = False
    if (k+1) in males:
        isMale = True
    df.loc[k] = score(row, tables, isMale)

df.to_csv(SAVE_NAME, encoding='utf-8', sep=',', index=False, header=False)
