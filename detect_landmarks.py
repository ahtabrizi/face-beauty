import face_alignment
from skimage import io
from plot_landmarks import *
import numpy as np 
from transforms import *
import os.path
    
# Run the 3D face alignment on a test image, without CUDA.
FA = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False)
TAG_NAME = "pre"
BASE_DIR = "data/" + TAG_NAME + '/'
for i in range(1,377):
    print("Processing Case #", i)

    # if i in [35, 36]:
    #     print("excluding current image due to errors!")
    #     continue

    SAVE_DIR = BASE_DIR + "data/landmarks#" + str(i) + ".npy"
    if os.path.exists(SAVE_DIR):
        # print("Data file already exists, moving on!")
        continue

    left_img = io.imread(BASE_DIR + 'Case#'+ str(i) + '_3_L_' + TAG_NAME + '_Pa_Ro_Sc.png')[180:900, :, :3]
    right_img = io.imread(BASE_DIR + 'Case#'+ str(i) + '_2_R_' + TAG_NAME + '_Pa_Ro_Sc.png')[180:900, :, :3]
    front_img = io.imread(BASE_DIR + 'Case#'+ str(i) + '_1_F_' + TAG_NAME + '_Pa_Ro_Sc_Tr_Ep.png')[240:750, 110:610, :3]

    # try:
    #     left_preds = np.asarray(FA.get_landmarks(left_img)[-1])
    #     right_preds = np.asarray(FA.get_landmarks(right_img)[-1])
    # except:
    #     print("excluding current image due to errors!")
    #     continue
    
    # Left view image
    left_not_detected = False 
    left_flipped = False 
    try:
        left_preds = np.asarray(FA.get_landmarks(left_img)[-1])
    except:
        try:
            left_preds = np.asarray(FA.get_landmarks(left_img[:, ::-1, :])[-1])
            left_preds = flip(left_preds, left_img.shape[1], 0)
            left_flipped = True
        except:
            left_not_detected = True
    # right view image
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
                print("excluding current image due to errors!")
                continue
            else:
                right_preds = left_preds.copy()
                if not left_flipped:
                    right_preds = flip(right_preds, right_img.shape[1], 0)
        
    front_preds =  np.asarray(FA.get_landmarks(front_img)[-1])    

    right_preds = flip(right_preds, right_img.shape[1], 0)
    right_preds = coincideLandmarkLR(left_preds, right_preds)

    # swap x and z axes to be same with frontal axes
    left_preds[:, [0, 2]] = left_preds[:, [2, 0]]
    right_preds[:, [0, 2]] = right_preds[:, [2, 0]]

    front_preds = flip(front_preds, np.max(front_preds[:,2]), 2)
    front_preds = coincideLandmarkFP(front_preds, left_preds)

    LRFP = mergeLandmarks(left_preds, right_preds, front_preds)

    np.save(SAVE_DIR, LRFP)
    del left_img, right_img,front_img

