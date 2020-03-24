import numpy as np 

def flip(array, width, index):
    result = np.asarray(array)
    result[:, index] = width - result[:, index] 
    return result

# calculate magnitude of vector
def mag(v):
    return np.sqrt(v.dot(v))

# skew symmetric matrix of a v
def skew(v):
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
    
def rotationMatrix(v1, v2):
    Raxis = np.cross(v1, v2, axis=0) /(mag(v1) * mag(v2))
    sint = mag(Raxis)
    cost = np.dot(v1, v2) /(mag(v1) * mag(v2))
    
    if np.abs(sint) < 0.01:
        return np.eye(3)
    
    K = skew(Raxis / sint)

    # using rodriguez formula (a bit modified)
    return np.eye(3) + sint * K + (1 - cost) * (K @ K)

def transformationMat(rot, trans):
    return np.vstack((np.hstack((rot, trans)), np.array([0,0,0,1])))
  
    
# coincides right_preds(RP) and left_preds(LP) using tip of the nose
# using LP's nose point as reference
def coincideLandmarkLR(LP, RP):
    nTop = 27 # index of the top point on nose
    nTip = 29 # index of tip of the nose in predictions
    dx, dy, dz = RP[nTip] - LP[nTip] 

    RP[:, 0] -= dx
    RP[:, 1] -= dy
    RP[:, 2] -= dz

    # rotate RP to coincide better with LP
    vR = RP[nTop] - RP[nTip]
    vL = LP[nTop] - LP[nTip]
    vR[2] = 0
    vL[2] = 0
    R = rotationMatrix(vR, vL)

    Trot = transformationMat(R, np.zeros((3,1)))
    Ttrans = transformationMat(np.eye(3) , np.expand_dims(RP[nTip].T, axis=1)) 
    nTtrans = transformationMat(np.eye(3) , -np.expand_dims(RP[nTip].T, axis=1)) 

    RPT = np.hstack((RP , np.ones((RP.shape[0], 1))))

    #RP = ((Ttrans @ Trot @ nTtrans) @ RPT.T)[:3, :].T
    RP = ( RPT @ (Ttrans @ Trot @ nTtrans).T)[:, :3]
   
    return RP

# coincides front_preds(RP) and profile_preds(LP) using tip of the nose
# using PP's nose point as reference
def coincideLandmarkFP(FP, PP):
    nTip = 29 # index of tip of the nose in predictions
    dx, dy, dz = FP[nTip] - PP[nTip] 

    FP[:, 0] -= dx
    FP[:, 1] -= dy
    FP[:, 2] -= dz

    return FP

#def mergeLR(LP, RP):
#    merged = []
#    # jaw
#    merged[:9] = RP[:9]
#    merged[9:17] = LP[7::-1]
#    merged[8] = (merged[8] + LP[8]) / 2 # middle point
#
#    # eyebrow
#    merged[17:22] = RP[17:22]
#    merged[22:27] = LP[17:22]
#
#    # nose
#    merged[27:31] = (RP[27:31] + LP[27:31]) / 2  
#
#    # under nose
#    merged[31:34] = RP[31:34]
#    merged[34:36] = LP[32:30:-1]
#    merged[33] = (merged[33] + LP[33]) / 2 # middle point
#
#    # eye
#    merged[36:42] = RP[36:42]
#    merged[42:48] = LP[36:42]
#
#    # lips
#    merged[48:68] = RP[48:68]
#    merged = np.asarray(merged)
#    i = [52, 53, 54, 55, 56, 63, 64, 65]
#    j = [50, 49, 48, 59, 58, 61, 60, 67]
#    merged[j] = LP[i]
#    i = [51, 57, 62, 66]
#    merged[i] = (merged[i] + LP[i]) / 2
#
#    return merged
#
def mergeLR(LP, RP):
    merged = np.zeros_like(LP)
    merged[:, 0] = LP[:, 0]
    merged[:, 1:] = (LP[:, 1:] + RP[:, 1:]) / 2
    return merged

def mergeLandmarks(LP, RP, FP):
    # merge left and right
    LRP = mergeLR(LP, RP)
    
    finalP = np.zeros_like(FP)
    finalP[:, 0] = FP[:, 0]
    finalP[:, 2] = LRP[:, 2]
    finalP[:, 1] = (FP[:, 1] + LRP[:, 1]) / 2

    return finalP

















