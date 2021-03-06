import face_alignment
from skimage import io
from plot_landmarks import *
import numpy as np 
from transforms import *

    
# Run the 3D face alignment on a test image, without CUDA.
FA = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False)

BASE_DIR = "./data/pre/"
i = 98 
left_img = io.imread(BASE_DIR + 'Case#'+ str(i) + '_3_L_pre_Pa_Ro_Sc.png')[180:900, :, :3]
right_img = io.imread(BASE_DIR + 'Case#'+ str(i) + '_2_R_pre_Pa_Ro_Sc.png')[180:900, :, :3]
front_img = io.imread(BASE_DIR + 'Case#'+ str(i) + '_1_F_pre_Pa_Ro_Sc_Tr_Ep.png')[240:750, 110:610, :3]
plt.imshow(right_img)
plt.figure()
plt.imshow(left_img)
plt.show()

left_not_detected = False 
left_flipped = False 
print("left")
try:
    left_preds = np.asarray(FA.get_landmarks(left_img)[-1])
except:
    try:
        left_preds = np.asarray(FA.get_landmarks(left_img[:, ::-1, :])[-1])
        left_preds = flip(left_preds, left_img.shape[1], 0)
        left_flipped = True
    except:
        left_not_detected = True
        # raise "Cannot find face in left image"
print("right")
try:
    right_preds = np.asarray(FA.get_landmarks(right_img)[-1])
    if left_not_detected:
        left_preds = right_preds.copy()
        left_preds = flip(left_preds, left_img.shape[1], 0)
except:
    try:
        right_preds = np.asarray(FA.get_landmarks(right_img[:, ::-1, :])[-1])
        if left_not_detected:
            left_preds = right_preds.copy()
        right_preds = flip(right_preds, right_img.shape[1], 0)
    except:
        if left_not_detected:
            raise "Cannot find face in right image"
        else:
            right_preds = left_preds.copy()
            if not left_flipped:
                right_preds = flip(right_preds, right_img.shape[1], 0)
    
print("front")
front_preds =  np.asarray(FA.get_landmarks(front_img)[-1])    

print("PLOTTING")

# Plot 2D on Face, left and Right 
fig = plt.figure()
ax = plt.gca()
ax.set_title("Left View")
drawLandmarksOnFace(ax, left_img, left_preds, color=(1,0,0))
fig = plt.figure()
ax = plt.gca()
ax.set_title("Right View")
drawLandmarksOnFace(ax, right_img, right_preds, color=(1,0,0))

right_preds = flip(right_preds, right_img.shape[1], 0)
right_preds = coincideLandmarkLR(left_preds, right_preds)

# Plot 2D
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

P = mergeLR(left_preds, right_preds)
LRFP = mergeLandmarks(left_preds, right_preds, front_preds)

# Plot 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
#drawLandmarks3D(ax, left_preds, label="Left")
#drawLandmarks3D(ax, right_preds, label="Right", cPoint=(1,0,0), cLine=(0.7,0.2,0.2))
#drawLandmarks3D(ax, P, label="LR",  cPoint=(1,1,0), cLine=(0.7,0.5,0.7))
drawLandmarks3D(ax, LRFP, label="LRFP")
drawLandmarks3D(ax, front_preds, label="Front",  cPoint=(0,1,0), cLine=(0.2,0.7,0.2))

#
#i = 0 
#ax.scatter(left_preds[i, 0], left_preds[i, 1],left_preds[i, 2], color=(0,1,1))
#ax.scatter(right_preds[i, 0], right_preds[i, 1],right_preds[i, 2],color=(0,1,0))
#
labelMaker(ax)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
