import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections



pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }

plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)
# 2D-Plot
def drawLandmarks2D(ax, preds, color=None) :
    for pred_type in pred_types.values():
        ax.plot(preds[pred_type.slice, 0],
                preds[pred_type.slice, 1],
                color=pred_type.color, **plot_style)
        if color is not None:
            ax.get_lines()[-1].set_color(color)

def drawLandmarksOnFace(ax, input_img, preds, color=None):
    ax.imshow(input_img)
    drawLandmarks2D(ax, preds, color)
    ax.axis('off')
    
# 3D-Plot
def drawLandmarks3D(ax , preds, cPoint='cyan', cLine='blue'):

    surf = ax.scatter(preds[:, 0] * 1.2,
                      preds[:, 1],
                      preds[:, 2],
                      color=cPoint,
                      alpha=1.0,
                      edgecolor='b')
    for pred_type in pred_types.values():
        ax.plot3D(preds[pred_type.slice, 0] * 1.2,
                  preds[pred_type.slice, 1],
                  preds[pred_type.slice, 2], color=cLine)

    ax.view_init(elev=89., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])

