# Disclaimer:
# This file is part of the undergraduate thesis of Mr. Efthimis Michalis.
# The thesis was developed under the supervision of Assistant Prof. Aggelos
# Pikrakis, in the Department of Informatics, School of ICT, University of
# Piraeus, Greece.


# METHODS.py
# Histogram Equalization
# [T0,T1] is sapce in histogram to which it will be applied the HE to NT0,NT1




def HE(img, T0=70, T1=170, NT0=170, NT1=255, Dark0=5, Dark1=50):
    import numpy as np
    from skimage import exposure
    import matplotlib.pyplot as plt
    L = np.linspace(0, 255, 256)
    img = np.array(img)
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 255))
    plt.bar(L, hist)
    plt.fill_between(
        L, 0, hist.max(),
        where=abs(L-(T1+T0)/2) <= (T1-T0)/2,
        facecolor='black',
        alpha=0.7,
        label=r'$Sum_{hist_{i}}\geqslant S_{shadow}$'
    )
    plt.plot(np.array([T0, T0]), np.array([0, hist.max()]), '--k')
    plt.plot(np.array([T1, T1]), np.array([0, hist.max()]), '--k')
    plt.fill_between(
        L, 0, hist.max(),
        where=abs(L-(Dark1+Dark0)/2) <= (Dark1-Dark0)/2,
        facecolor='#178D36',
        alpha=0.7,
        label=r'$Sum_{hist_{i}}\geqslant S_{dark}$'
    )

    plt.plot(np.array(
        [Dark0, Dark0]),
        np.array([0, hist.max()]),
        '--',
        color='#178D36'
    )

    plt.plot(
        np.array([Dark1, Dark1]),
        np.array([0, hist.max()]),
        '--',
        color='#178D36'
    )

    plt.legend()
    plt.xlabel(r'$Pixel_i$', fontsize=16)
    plt.ylabel(r'$population_i$', fontsize=16)
    plt.show()
    plt.bar(L, hist)
    plt.fill_between(
        L, 0, hist.max(),
        where=abs(L-(T1+T0)/2) <= (T1-T0)/2,
        facecolor='red',
        alpha=0.7,
        label=r'$[T_{0},T_{1}]$'
    )
    plt.plot(np.array([T0, T0]), np.array([0, hist.max()]), '--r')
    plt.plot(np.array([T1, T1]), np.array([0, hist.max()]), '--r')
    # NT0,NT1
    plt.fill_between(
        L, 0, hist.max(),
        where=abs(L-(NT1+NT0)/2) <= (NT1-NT0)/2,
        facecolor='Green',
        alpha=0.7,
        label=r'$[NT_{0},NT_{1}]$'
    )
    plt.plot(np.array([NT0, NT0]), np.array([0, hist.max()]), '--g')
    plt.plot(np.array([NT1, NT1]), np.array([0, hist.max()]), '--g')
    plt.xlabel(r'$Pixel_i$', fontsize=16)
    plt.ylabel(r'$population_i$', fontsize=16)
    plt.legend()
    plt.show()
    Dsum = np.sum(hist[Dark0:Dark1])
    Ssum = np.sum(hist[T0:T1+1])
    if(Dsum < img.size*0.18) or Ssum < img.size*0.34:
        print("Image does not meet the criteria for HE")
        return img, False
    hist = hist[T0:T1+1]
    L = np.linspace(T0, T1, T1-T0+1)
    hist = hist/img.size
    s = hist[0]

    for i in range(1, T1-T0+1):
        s += hist[i]
        hist[i] = s

    hist = (NT1-NT0)*hist+NT0
    hist = hist.astype(int)
    img2 = np.array(img)
    for i in range(0, T1-T0+1):
        B = np.where(img2 == (i+T0))
        img[B] = hist[i]

    plt.plot(L, hist)
    plt.xlabel(r'$OLDpixel_i$', fontsize=16)
    plt.ylabel(r'$NEWpixel_i$', fontsize=16)
    plt.title('Pixels update', fontsize=16)
    plt.show()
    L = np.linspace(0, 255, 256)
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 255))
    plt.bar(L, hist)
    plt.xlabel(r'$Pixel_i$', fontsize=16)
    plt.ylabel(r'$population_i$', fontsize=16)
    plt.title('Histogram after Histogram Equalization', fontsize=16)
    plt.show()
    return img, True


# s: sigma input for gaussin filter
# Tw: Weak Threshold
# Ts: Strong Threshold
def Cany_skimage(img, s=3, Tw=-1, Ts=-1):
    import numpy as np
    from skimage import filters as FL
    from skimage import feature
    # feature.canny calculate gaussin filter for img,soobel operator and
    # canny operator ,return a binary image
    if Tw == -1 or Ts == -1:
        Eimg = feature.canny(img, s)
    else:
        Eimg = feature.canny(img, s, Tw, Ts)
    # find continuous angle
    thita = np.arctan2(FL.sobel_v(img), FL.sobel_h(img))
    # zero array with thita's shape
    Ta = np.zeros((thita.shape[0], thita.shape[1]))

    # angle quantization
    B0 = np.where(np.logical_and(thita >= -1/8*np.pi, thita < np.pi/8))
    Ta[B0] = 1
    B1 = np.where(np.logical_or(thita >= 7/8*np.pi, thita < -7/8*np.pi))
    Ta[B1] = -1

    B0 = np.where(np.logical_and(thita >= np.pi/8, thita < 3/8*np.pi))
    Ta[B0] = 2
    B1 = np.where(np.logical_and(thita >= -7/8*np.pi, thita < -5/8*np.pi))
    Ta[B1] = -2

    B0 = np.where(np.logical_and(thita >= 3/8*np.pi, thita < 5/8*np.pi))
    Ta[B0] = 3
    B1 = np.where(np.logical_and(thita >= -5/8*np.pi, thita < -3/8*np.pi))
    Ta[B1] = -3

    B0 = np.where(np.logical_and(thita >= 5/8*np.pi, thita < 7/8*np.pi))
    Ta[B0] = 4
    B1 = np.where(np.logical_and(thita >= -3/8*np.pi, thita < -1/8*np.pi))
    Ta[B1] = -4

    A = np.where(Eimg < 1)
    Ta[A[0], A[1]] = 0
    return Eimg, Ta


# 1) Find clusters with DBSCAN
# 2) per cluster find and return the corner /
# ## points of rectangle that is contained
def DBs(Pon):
    import numpy as np
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt
    Rect = []
    # All coordinates are intergers and we need square area around the points
    Eps = np.sqrt(2)+0.001
    Min_samples = 2
    db = DBSCAN(Eps, Min_samples).fit(Pon)  # Uses euclidean distance
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    for k in zip(unique_labels):
        class_member_mask = (labels == k)
        xy = Pon[class_member_mask & core_samples_mask]
        if xy.size > 0:
            Rect.append([
                np.min(xy[:, 0]),  # posx0
                np.min(xy[:, 1]),  # posy1
                np.max(xy[:, 0]),  # posx1
                np.max(xy[:, 1])   # posy1
            ])
    return np.array(Rect).reshape(len(Rect), 4)


# Rect is <n,4> array like DBs return value
# heightCluster return Rect Clusters which have /
# almost the same posx0a & posx1, that means same height and position
def heightCluster(Rect, Esp=15, Min_samples=4):
    import numpy as np
    import matplotlib.pyplot as plt
    K = [0, 2]  # 0 -> posx0 , 2-> posx 1
    Clusts = np.array([], dtype=np.int).reshape(0, 4)
    poni = Rect[0]
    PointsToNext = 0
    plt.plot(Rect[:, 0], Rect[:, 2], 'o', Label='Other')
    Fx = np.array([Rect[:, [0]].min(), Rect[:, [2]].max()])
    Rect = np.delete(Rect, 0, 0)
    while Rect.shape[0] > 0:
        # Euclidean distance
        A = np.sqrt((
            poni[K[0]]-Rect[:, K[0]])**2+(poni[K[1]]-Rect[:, K[1]])**2
        ) <= Esp

        if np.count_nonzero(A) >= Min_samples:
            PointsToNext = Rect[A]
            Clusts = np.concatenate((Clusts, poni.reshape(1, 4), PointsToNext))
            A = np.where(A)
            Rect = np.delete(Rect, A, 0)
            while PointsToNext.shape[0] > 0:
                Pi = PointsToNext[0]
                PointsToNext = np.delete(PointsToNext, 0, 0)

                A = np.sqrt((
                    Pi[K[0]]-Rect[:, K[0]])**2+(Pi[K[1]]-Rect[:, K[1]])**2
                ) <= Esp

                if np.count_nonzero(A) >= Min_samples:
                    A = np.where(A)
                    Np = Rect[A]
                    PointsToNext = np.concatenate((PointsToNext, Np))
                    Rect = np.delete(Rect, A, 0)
                    Clusts = np.concatenate((Clusts, Np))
        else:
            poni = Rect[0]
            Rect = np.delete(Rect, 0, 0)

    plt.xlabel(r'$PosY_{min}$', fontsize=16)
    plt.ylabel(r'$PosY_{max}$', fontsize=16)
    plt.plot(Clusts[:, 0], Clusts[:, 2], 'or', label='Possible characters')
    plt.plot(Fx, Fx, '--')
    plt.title('Density = '+str(Min_samples)+', Radius = '+str(Esp), fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    return Clusts

# maxA max Area percentage
# minh min height percentage
# remove all very large and small Rect and merge overlap Rects


def AreaCluster(Rect, Shape, maxA=0.1, minh=0.1):
    import numpy as np
    Rect = list(Rect)
    Rect2 = []
    A = Shape[0]*Shape[1]*maxA
    H = Shape[1]*minh
    for i in range(len(Rect)):
        posx0 = Rect[i][0]
        posy0 = Rect[i][1]
        posx1 = Rect[i][2]
        posy1 = Rect[i][3]

        # checing for small and large Rects
        if (posx1-posx0)*(posy1-posy0) < A and (posx1-posx0)*(posy1-posy0) > 0.05*A and (posx1-posx0) > H:
            j = 0
            Bool = True

            # merge overlap Rects
            while Bool and j < len(Rect2):
                # overlap operation
                if (posy0-Rect2[j][1])*(posy1-Rect2[j][3]) <= 0 and (posx0-Rect2[j][0])*(posx1-Rect2[j][2]) <= 0:
                    Bool = False
                    Rect2[j][0] = min(posx0, Rect2[j][0])
                    Rect2[j][1] = min(posy0, Rect2[j][1])
                    Rect2[j][2] = max(Rect2[j][2], posx1)
                    Rect2[j][3] = max(Rect2[j][3], posy1)
                j += 1
            if Bool:
                Rect2.append([posx0, posy0, posx1, posy1])
    Rect = Rect2
    return np.array(Rect).reshape(len(Rect), 4)


# draw Rects on the image
# c :rgb list [r,g,b] 0 to 1 range
def RectsDrawing(ImP, Rect, c):
    import matplotlib.pyplot as plt
    import numpy as np
    ImP = np.array(ImP)
    n_clusters_ = Rect.shape[0]
    for I in range(0, n_clusters_):
        pos0x = Rect[I][0]
        pos0y = Rect[I][1]
        pos1x = Rect[I][2]
        pos1y = Rect[I][3]
        plt.plot(
            [pos0y, pos1y, pos1y, pos0y, pos0y],
            [pos0x, pos0x, pos1x, pos1x, pos0x],
            '--b'
        )
    plt.imshow(ImP)
