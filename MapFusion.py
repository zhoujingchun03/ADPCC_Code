import numpy as np
import math
import cv2
e = math.e

def S(a,sigma):
    FinalS = (1 + e ** (-32 * (a - sigma))) ** -1
    return FinalS

def S1(img,sigma):
    height = img.shape[0]
    width = img.shape[1]
    Filter_more_half = []
    for i in range(height):
        for j in range(width):
            if(img[i,j]>(0.5*255)):
                Filter_more_half.append(img[i,j])
    Length_more_half = len(Filter_more_half)
    a = Length_more_half/(height * width)
    FinalS = (1 + e ** (-32 * (a - sigma))) ** -1
    return FinalS


def Scene_depth(d_R,d_D,img):
    I_gray_Q = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alpha = S1(I_gray_Q, sigma=0.2)
    # print('alpha：',alpha)

    Depth_map =alpha * d_D  +  (1  - alpha) *  d_R
    return Depth_map

def flag(img):
    avg_Ib = np.mean(img[:,:,0])
    # beta = S(2, k)
    beta = S(avg_Ib, 0.3*255)
    print('beta：', beta)
    return beta

def Scene_depth_fusion(mip_r,mono2,beta):
    Depth_map =beta * mono2  +  (1  - beta) *  mip_r
    return Depth_map
