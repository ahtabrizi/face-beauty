import face_alignment
from skimage import io
from plot_landmarks import *

# Run the 3D face alignment on a test image, without CUDA.
FA = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False)

left_img = io.imread('post/Case#1_3_L_post.png')
right_img = io.imread('post/Case#1_2_R_post.png')

left_preds = FA.get_landmarks(left_img)[-1]
right_preds = FA.get_landmarks(right_img)[-1]



#fig = plt.figure(figsize=plt.figaspect(.5))
drawLandmarksOnFace(left_img, left_preds)
drawLandmarks2D(plt.gca(), right_preds)
print(right_preds)
#plotLandmarks3D(preds, fig)
plt.show()
