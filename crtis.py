import numpy as np 
from transforms import mag
from plot_landmarks import *

def crit1(preds):
    A = preds[36, 0]
    B = preds[39, 0]
    C = preds[42, 0]
    D = preds[45, 0]

    AB = B - A
    BC = C - B
    CD = C - D
    
    AD = AB + BC + CD

    return AB/AD, BC/AD, CD/AD

def crit3(preds):
    A = preds[8, 1]
    B = (preds[62, 1] + preds[66, 1]) /2
    C = preds[33, 1]
    D = (preds[21, 1] + preds[22, 1]) /2

    AB = A - B
    BC = B - C
    CD = C - D

    return AB/BC, (AB+BC)/CD

def crit4(preds):
    p1 = (preds[35, 2] + preds[31, 2]) / 2 # B
    p2 = preds[29, 2] # A
    p3 = preds[51, 2] # O

    AO = p2 - p3
    AB = p2 - p1
    return AO / AB

def crit5(preds):
    p1 = (preds[35, 2] + preds[31, 2]) / 2 # B
    p2 = preds[29, 2] # O
    p3 = preds[27, 2] # A

    OB = p2 - p1
    OA = p2 - p3

    return OB / OA

def crit6(preds):
    p1 = preds[27, 1:]
    p2 = preds[30, 1:]
    p3 = (preds[22, 1:] + preds[21, 1:]) /2
    p4 = preds[35, 1:]

    v1 = p2 - p1
    v2 = p4 - p3

    return np.degrees(np.arccos(np.dot(v1, v2) / (mag(v1) * mag(v2))))

def crit7(preds):
    p1 = preds[33, 1:]
    p2 = preds[30, 1:]
    p3 = (preds[50, 1:] + preds[52, 1:]) /2

    v1 = p2 - p1
    v2 = p3 - p1

    return np.degrees(np.arccos(np.dot(v1, v2) / (mag(v1) * mag(v2))))


def crit8(preds):
    p1 = preds[27, 1:]
    p2 = preds[30, 1:]
    p3 = (preds[22, 1:] + preds[21, 1:]) /2

    v1 = p2 - p1
    v2 = p3 - p1

    return np.degrees(np.arccos(np.dot(v1, v2) / (mag(v1) * mag(v2))))

def lips(preds):
    A = preds[48, 0]
    B = preds[54, 0]
    C = preds[57, 1]
    D = preds[50, 1]

    AB = A - B
    DC = D - C
    print(AB, DC)

    return DC / AB
LRFP = np.load("post/data/landmarks#333.npy")
print(lips(LRFP))
fig = plt.figure()
ax = plt.axes(projection='3d')

drawLandmarks3D(ax, LRFP)

plt.show()
