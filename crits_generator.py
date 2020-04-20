import numpy as np
import pandas as pd
import glob
from crits import *

males = [4, 12, 13, 14, 17, 31, 47, 48, 56, 57, 61, 71, 72, 77, 92, 101,
103, 109, 111, 121, 133, 153, 161, 163, 164, 167, 170, 179, 190, 198,
201, 205, 222, 229, 239, 251, 252, 255, 271, 272, 285, 310, 318, 347, 354, 369]
for TAG_NAME in ["pre", "post"]:
    SAVE_NAME = "results/Quantitative_" + TAG_NAME + ".csv"


    paths = []
    paths.extend(glob.glob("data/" + TAG_NAME + "/data/*.npy"))

    
    df = pd.DataFrame(columns=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    for i, path in enumerate(paths):
        isMale = False
        if i in males:
            isMale = True
        LRFP = np.load(path)
        df.loc[i] = [ pcrit1(LRFP),
                      0,
                      pcrit3(LRFP),
                      pcrit4(LRFP),
                      pcrit5(LRFP),
                      pcrit6(LRFP),
                      pcrit7(LRFP, isMale),
                      pcrit8(LRFP),
                      0]
    

    df.to_csv(SAVE_NAME, encoding='utf-8', sep=',', index=False)

