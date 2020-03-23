import face_alignment
from skimage import io
from plot_landmarks import *
import numpy as np 

# Run the 3D face alignment on a test image, without CUDA.
FA = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False)

left_img = io.imread('post/Case#1_3_L_post.png')
right_img = io.imread('post/Case#1_2_R_post.png')
print(right_img.shape)

left_preds = np.asarray(FA.get_landmarks(left_img)[-1])
right_preds = np.asarray(FA.get_landmarks(right_img)[-1])


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
    
right_preds = flipPreds(right_preds, right_img.shape[1])
right_preds = coincideLandmarkLR(left_preds, right_preds)

#fig = plt.figure(figsize=plt.figaspect(.5))
#drawLandmarksOnFace(left_img, left_preds, color=(1,0,0))
ax = plt.gca()
drawLandmarks2D(ax, left_preds, color=(1,0,0))
drawLandmarks2D(ax, right_preds, color=(0,0,1))
#plotLandmarks3D(preds, fig)

i = 27
#ax.plot(left_preds[i, 0], left_preds[i, 1],color=(0,1,0), **plot_style)
ax.set_xlim([0, right_img.shape[1]])
ax.set_ylim([0, right_img.shape[0]])
ax.set_aspect('equal')
ax.invert_yaxis()
plt.show()
