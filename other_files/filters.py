# Disclaimer:
# This file is part of the undergraduate thesis of Mr. Efthymis Michalis.
# The thesis was developed under the supervision of Assistant Prof. Aggelos
# Pikrakis, in the Department of Informatics, School of ICT, University of
# Piraeus, Greece.

# FILTERS.py
import numpy as np
from scipy.ndimage.filters import gaussian_filter
# Example gaussian_filter(Input,s=2.7)


# return a square scaled image
def ScaleArray(Input, ScaleSize, Type='bicubic'):
    from scipy.misc import imresize
    return imresize(Input, (ScaleSize, ScaleSize), Type)/255


def Normalize(Input):
        return 1/(Input.max()-Input.min())*(Input-Input.min())


def Thresholding(Input, Tmin=0, Tmax=1, Type='binary'):
    Thres = (np.sign(Input-Tmin)+np.sign(Tmax-Input))/2
    if Type == 'binary':
        return Thres
    elif Type == 'linear':
        return Thres*Input


def GetPoints(Input, T=0.45):
    Ponts = np.where(Input > T)
    P1 = Ponts[1].reshape(Ponts[1].shape[0], 1)
    P0 = Ponts[0].reshape(Ponts[0].shape[0], 1)
    return np.concatenate((P0, P1), axis=1)


# A blur filter
# k     : karnel_size odd number
# s     : sigma
def GaussFilter(Input, k=5, s=2):
    if(k % 2 == 0):
        print("k must be odd number")
        k += 1
        print("New k = %i" % (k))
    x = (np.arange(k)-(k-1)/2).reshape(1, k)
    y = x.transpose()
    z = -(y**2+x**2)/(s**2*2)
    z = 1/(2*s**2*np.pi)*np.exp(z)
    Mask = z/np.sum(z)
    return Use_Mask(Input, Mask)


# START EDGE FILTERS
def Use_Mask(Input, Mask):
        A = []
        H = Input.shape[0]
        W = Input.shape[1]
        k = Mask.shape[0]
        ki = k//2
        Input = np.concatenate((
            Input[:, 0].reshape(H, 1)*np.ones((H, ki)),
            Input, Input[:, -1].reshape(H, 1)*np.ones((H, ki))
           ), axis=1)
        Input = np.concatenate((
                Input[0, :].reshape(1, W+2*ki)*np.ones((ki, W+2*ki)),
                Input,
                Input[-1, :].reshape(1, W+2*ki)*np.ones((ki, W+2*ki))
                ), axis=0)
        for i in range(0, H):
            for j in range(0, W):
                A.append(np.sum(Input[i:i+k, j:j+k]*Mask))
        return np.array(A).reshape(H, W)


# Laplacian of Gaussian ###
# input : array
# k     : karnel_size odd number
# s     : sigma
def LoG(Input, k=7, s=1.4):
    if(k % 2 == 0):
        print("k must be odd number")
        k += 1
        print("New k = %i" % (k))
    Min = int(-(k-1)/2)
    Max = -Min
    x = (np.linspace(Min, Max, k)**2).reshape(1, k)
    y = x.transpose()
    z = (y+x)/(s**2*2)
    LoG = 1/(s**4*np.pi)*(z-1)*np.exp(-z)
    return Use_Mask(Input, LoG)


def Roberts(Input):
    #      __     __        __     __  #
    #     | +1    0 |      | 0    +1 | #
    # Gx= |         | ,Gy= |         | #
    #     | 0    -1 |      | -1    0 | #
    #      ‾‾     ‾‾        ‾‾     ‾‾  #
    A = []
    Ponts = []
    H = Input.shape[0]
    W = Input.shape[1]
    for i in range(0, H-1):
        for j in range(0, W-1):
            A.append(np.sqrt((np.sqrt(Input[i, j])-np.sqrt(Input[i+1, j+1]))**2+(np.sqrt(Input[i+1, j])-np.sqrt(Input[i, j+1]))**2))
    return np.array(A).reshape(H-1, W-1)


def Kirsch(Input):
    Mask_Kir = np.array([
        [
            [ 5,  5,  5],
            [-3,  0, -3],
            [-3, -3, -3]
        ],
        [
            [-3,  5,  5],
            [-3,  0,  5],
            [-3, -3, -3]
        ],
        [
            [-3,  -3,  5],
            [-3,   0,  5],
            [-3,  -3,  5]
        ],
        [
            [-3, -3, -3],
            [-3,  0,  5],
            [-3,  5,  5]
        ],
        [
            [-3, -3, -3],
            [-3,  0, -3],
            [ 5,  5,  5]
        ],
        [
            [-3, -3, -3],
            [5,   0, -3],
            [5,   5, -3]
        ],
        [
            [5, -3, -3],
            [5,  0, -3],
            [5, -3, -3]
        ],
        [
            [ 5,  5, -3],
            [ 5,  0, -3],
            [-3, -3, -3]
        ]
    ])
    return Use_Mask(Input, Mask_Kir)


def Sobel(Input):
    Mask_S = np.array([
        [
            [ 1,  2,  1],
            [ 0,  0,  0],
            [-1, -2, -1]
        ],
        [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]
    ])
    return Use_Mask(Input, Mask_S)


def Prewitt(Input):
    Mask_Pr = np.array([
        [
            [ 1,  1,  1],
            [ 0,  0,  0],
            [-1, -1,  -1]
        ],
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]
    ])
    return Use_Mask(Input, Mask_Pr)
# END EDGE FILTERS


# CannyOprt
# CannyOprt
# Ts : Strong threshold
# Tw : Weak threshold
def SimpleCannyOperator(img, Ts=0.3, Tw=0.1):
    H = img.shape[0]
    W = img.shape[1]
    img = np.concatenate((img, np.zeros((H, 1))), axis=1)
    img = np.concatenate((img, np.zeros((1, W+1))), axis=0)
    Edg = []
    Thita = []
    for i in range(0, H):
        for j in range(0, W):

            # Robert Edge detection
            g1 = img[i, j]-img[i+1, j+1]
            g2 = img[i, j+1]-img[i+1, j]
            Gx = g1-g2
            Gy = g1+g2
            Edg.append(Gx**2+Gy**2)
            Thita.append(np.arctan2(Gx, Gy))

            # angle quantization
            if (Thita[-1] >= -1/8*np.pi and Thita[-1] < np.pi/8):
                Thita[-1] = 1
            elif (Thita[-1] >= 7/8*np.pi or Thita[-1] < -7/8*np.pi):
                Thita[-1] = -1
            elif (Thita[-1] >= np.pi/8 and Thita[-1] < 3/8*np.pi):
                Thita[-1] = 2
            elif (Thita[-1] >= -7/8*np.pi and Thita[-1] < -5/8*np.pi):
                Thita[-1] = -2
            elif (Thita[-1] >= 3/8*np.pi and Thita[-1] < 5/8*np.pi):
                Thita[-1] = 3
            elif (Thita[-1] >= -5/8*np.pi and Thita[-1] < -3/8*np.pi):
                Thita[-1] = -3
            elif (Thita[-1] >= 5/8*np.pi and Thita[-1] < 7/8*np.pi):
                Thita[-1] = 4
            elif (Thita[-1] >= -3/8*np.pi and Thita[-1] < -1/8*np.pi):
                Thita[-1] = -4
    Thita = np.array(Thita).reshape(H, W)
    Edg = np.sqrt(np.array(Edg).reshape(H, W))
    Edg = np.concatenate((np.zeros((H, 1)), Edg, np.zeros((H, 1))), axis=1)
    Edg = np.concatenate((np.zeros((1, W+2)), Edg, np.zeros((1, W+2))), axis=0)

    # Remove non local maximum magnitude
    for i in range(1, H):
        for j in range(1, W):
            if abs(Thita[i-1, j-1]) == 1:
                if Edg[i, j] >= Edg[i-1, j] and Edg[i, j] >= Edg[i+1, j]:
                    Edg[i-1, j] = 0
                    Edg[i+1, j] = 0
                else:
                    Edg[i, j] = 0
            elif abs(Thita[i-1, j-1]) == 2:
                if Edg[i, j] >= Edg[i-1, j-1] and Edg[i, j] >= Edg[i+1, j+1]:
                    Edg[i-1, j-1] = 0
                    Edg[i+1, j+1] = 0
                else:
                    Edg[i, j] = 0
            elif abs(Thita[i-1, j-1]) == 3:
                if Edg[i, j] >= Edg[i, j-1] and Edg[i, j] >= Edg[i, j+1]:
                    Edg[i, j-1] = 0
                    Edg[i, j+1] = 0
                else:
                    Edg[i, j] = 0
            elif abs(Thita[i-1, j-1]) == 4:
                if Edg[i, j] >= Edg[i-1, j+1] and Edg[i, j] >= Edg[i+1, j-1]:
                    Edg[i-1, j+1] = 0
                    Edg[i+1, j-1] = 0
                else:
                    Edg[i, j] = 0

    # hysteresis thresholding
    Edg[np.where(Edg < Tw)] = 0
    Ponts = np.where(Edg >= Ts)
    Edg[Ponts] = 1
    P1 = Ponts[1].reshape(Ponts[1].shape[0], 1)
    P0 = Ponts[0].reshape(Ponts[0].shape[0], 1)
    PonS = np.concatenate((P0, P1), axis=1)

    Ponts = np.where(np.logical_and(Edg < 1, Edg > 0))
    P1 = Ponts[1].reshape(Ponts[1].shape[0], 1)
    P0 = Ponts[0].reshape(Ponts[0].shape[0], 1)
    PonC = np.concatenate((P0, P1), axis=1)

    X1 = np.array(PonS)
    poni = PonS[0]
    PonS = np.delete(PonS, 0, 0)
    while PonS.shape[0] > 0:
        As = (poni-1-PonS)*(poni+1-PonS) <= 0
        Ac = (poni-1-PonC)*(poni+1-PonC) <= 0

        As = np.where(np.logical_and(As[:, 0], As[:, 1]))
        Ac = np.where(np.logical_and(Ac[:, 0], Ac[:, 1]))
        PointsToNext = PonS[As]
        PointsToNext = np.concatenate((PointsToNext, PonC[Ac]))
        if PointsToNext.shape[0] > 0:
            PonS = np.delete(PonS, As, 0)
            PonC = np.delete(PonC, Ac, 0)
            while PointsToNext.shape[0] > 0:
                Pi = PointsToNext[0]
                PointsToNext = np.delete(PointsToNext, 0, 0)
                As = (Pi-1-PonS)*(Pi+1-PonS) <= 0
                Ac = (Pi-1-PonC)*(Pi+1-PonC) <= 0
                As = np.where(np.logical_and(As[:, 0], As[:, 1]))
                Ac = np.where(np.logical_and(Ac[:, 0], Ac[:, 1]))
                Np = PonS[As]
                Np = np.concatenate((Np, PonC[Ac]))
                if Np.shape[0] > 0:
                    PointsToNext = np.concatenate((PointsToNext, Np))
                    PonS = np.delete(PonS, As, 0)
                    PonC = np.delete(PonC, Ac, 0)
        else:
            poni = PonS[0]
            PonS = np.delete(PonS, 0, 0)
    Edg[PonC[:, 0], PonC[:, 1]] = 0
    Edg[np.where(Edg > 0)] = 1
    Edg = Edg[2:-2, 2:-2]
    Thita = Thita[1:-1, 1:-1]
    Thita[np.where(Edg < 1)] = 0
    return Edg, Thita
