import face_alignment
from skimage import io
from plot_landmarks import *
import numpy as np 
from transforms import *

# Run the 3D face alignment on a test image, without CUDA.
FA = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False)

left_img = io.imread('post/Case#1_3_L_post_Pa_Ro.png')
right_img = io.imread('post/Case#1_2_R_post_Pa_Ro.png')
front_img = io.imread('post/Case#1_1_F_post_Pa_Ro_Sc_Tr_Ep.png')

left_preds = np.asarray(FA.get_landmarks(left_img)[-1])
right_preds = np.asarray(FA.get_landmarks(right_img)[-1])
front_preds =  np.asarray(FA.get_landmarks(front_img)[-1])    

print("PLOTTING")
fig = plt.figure()
ax = plt.gca()
drawLandmarksOnFace(ax, left_img, left_preds, color=(1,0,0))
fig = plt.figure()
ax = plt.gca()
drawLandmarksOnFace(ax, right_img, right_preds, color=(1,0,0))

right_preds = flip(right_preds, right_img.shape[1], 0)
right_preds = coincideLandmarkLR(left_preds, right_preds)

fig = plt.figure()
ax = plt.axes()
drawLandmarks2D(ax, left_preds, color=(0,0,1))
drawLandmarks2D(ax, right_preds, color=(1,0,0))
#i = 27
#ax.plot(left_preds[i, 0], left_preds[i, 1],color=(0,1,0), **plot_style)
ax.set_xlim([0, right_img.shape[1]])
ax.set_ylim([0, right_img.shape[0]])
ax.set_aspect('equal')
ax.invert_yaxis()

# swap x and z axes to be same with frontal axes
left_preds[:, [0, 2]] = left_preds[:, [2, 0]]
right_preds[:, [0, 2]] = right_preds[:, [2, 0]]

front_preds = flip(front_preds, np.max(front_preds[:,2]), 2)
front_preds = coincideLandmarkFP(front_preds, left_preds)

fig = plt.figure()
ax = plt.axes(projection='3d')
drawLandmarks3D(ax, left_preds, label="Left")
drawLandmarks3D(ax, right_preds, label="Right", cPoint=(1,0,0), cLine=(0.7,0.2,0.2))
drawLandmarks3D(ax, front_preds, label="Front",  cPoint=(0,1,0), cLine=(0.2,0.7,0.2))

labelMaker(ax)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
