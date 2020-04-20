import numpy as np 
from transforms import mag
from plot_landmarks import *

# crit -> get proportions 
# pcrit -> get and score proportions 

def crit1(preds):
    A = preds[36, 0]
    B = preds[39, 0]
    C = preds[42, 0]
    D = preds[45, 0]

    AB = B - A
    BC = C - B
    CD = D - C
    
    AD = AB + BC + CD

    return AB/AD, BC/AD, CD/AD

def crit3(preds):
    A = preds[8, 1]                         # chin
    B = (preds[62, 1] + preds[66, 1]) /2    # middle of lips
    C = preds[33, 1]                        # nose base
    D = (preds[21, 1] + preds[22, 1]) /2    # middle of eyebrows

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
    p2 = preds[29, 2] # O - tip of the nose
    p3 = preds[27, 2] # A - top point of nose

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
    p1 = preds[27, 1:] # nose top point 
    p2 = preds[29, 1:] # nose tip
    p3 = (preds[22, 1:] + preds[21, 1:]) /2 # middle of eyebrows

    v1 = p2 - p1
    v2 = p3 - p1

    # add an offset to compensate for difference between actual middle of eyebrows
    return np.degrees(np.arccos(np.dot(v1, v2) / (mag(v1) * mag(v2)))) - 10

def lips(preds):
    A = preds[48, 0]
    B = preds[54, 0]
    C = preds[57, 1]
    D = preds[50, 1]

    AB = A - B
    DC = D - C
    print(AB, DC)

    return DC / AB

def pcrit1(preds):
    c = np.asarray(crit1(preds))
    diff = np.abs(c - 0.33)
    m = np.max(diff)
    if m < 0.02:
        return 10
    elif m < 0.12:
        return 7.5
    else:
        return 5
    
def pcrit3(preds):
    c = crit3(preds)
    diff = np.abs(c - np.array([1, 2]))
    m = np.max(diff)
    if m < 0.1:
        return 10
    elif m < 0.6:
        return 7.5
    else:
        return 5

def pcrit4(preds):
    c = crit4(preds)
    m = np.abs(c - 0.55)
    if m < 0.05:
        return 10
    elif m < 0.15:
        return 7.5
    else:
        return 5

def pcrit5(preds):
    c = crit5(preds)
    m = np.abs(c - 0.67)
    if m < 0.02:
        return 10
    elif m < 0.12:
        return 7.5
    else:
        return 5

def pcrit6(preds):
    c = crit6(preds)
    m = np.abs(c - 36)
    if m < 2:
        return 10
    elif m < 7:
        return 7.5
    else:
        return 5

def pcrit7(preds, isMale=False):
    c = crit7(preds)

    if isMale:
        m = np.abs(c - 95)
    m = np.abs(c - 105)

    if m < 5:
        return 10
    elif m < 15:
        return 7.5
    else:
        return 5

def pcrit8(preds):
    c = crit8(preds)
    m = np.abs(c - 105)
    if m < 5:
        return 10
    elif m < 15:
        return 7.5
    else:
        return 5


# LRFP = np.load("data/post/data/landmarks#335.npy")
# a= [ crit1(LRFP),
#    0,
#    crit3(LRFP),
#    crit4(LRFP),
#    crit5(LRFP),
#    crit6(LRFP),
#    crit7(LRFP),
#    crit8(LRFP),
#    0]
# b= [ pcrit1(LRFP),
#    0,
#    pcrit3(LRFP),
#    pcrit4(LRFP),
#    pcrit5(LRFP),
#    pcrit6(LRFP),
#    pcrit7(LRFP),
#    pcrit8(LRFP),
#    0]
# print(a)
# print(b)
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# drawLandmarks3D(ax, LRFP)

# plt.show()
