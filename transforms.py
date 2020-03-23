import numpy as np 

def flipPreds(preds, width):
    result = np.asarray(preds)
    result[:, 0] = width - result[:, 0] 
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
    dx, dy, _ = RP[nTip] - LP[nTip] 

    RP[:, 0] -= dx
    RP[:, 1] -= dy

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

