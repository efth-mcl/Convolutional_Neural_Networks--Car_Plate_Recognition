# Disclaimer:
# This file is part of the undergraduate thesis of Mr. Efthimis Michalis.
# The thesis was developed under the supervision of Assistant Prof. Aggelos
# Pikrakis, in the Department of Informatics, School of ICT, University of
# Piraeus, Greece.

#ALL_CAR_PLATES_TESTING_RESULTS.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as spm
import sys
import time
from skimage import exposure
from skimage import feature
from skimage import filters as FL
from sklearn.cluster import DBSCAN
sys.path.append('../LIBRARY')
from NEURAL_NETWORK_TOOL import *



####################
def HE(img,T0=70,T1=170,NT0=170,NT1=255,Dark0=5,Dark1=50):
    L=np.linspace(0,255,256)
    img=np.array(img)
    hist,_=np.histogram(img.ravel(),bins=256,range=(0, 255))
    Dsum=np.sum(hist[Dark0:Dark1])
    Ssum=np.sum(hist[T0:T1+1])
    if(Dsum<img.size*0.18) or Ssum<img.size*0.34:

        return img
    hist=hist[T0:T1+1]
    L=np.linspace(T0,T1,T1-T0+1)
    hist=hist/img.size
    s=hist[0]

    for i in range(1,T1-T0+1):
        s+=hist[i]
        hist[i]=s

    hist=(NT1-NT0)*hist+NT0
    hist=hist.astype(int)
    img2=np.array(img)
    for i in range(0,T1-T0+1):
        B=np.where(img2==(i+T0))
        img[B]=hist[i]
    return img
def Cany_skimage(img,s=3,Tw=-1,Ts=-1):

    if Tw==-1 or Ts==-1:
        Eimg=feature.canny(img,s)
    else:
        Eimg=feature.canny(img,s,0.1,0.3)

    thita=np.arctan2(FL.sobel_v(img),FL.sobel_h(img))
    Ta=np.zeros((thita.shape[0],thita.shape[1]))
    B0=np.where(np.logical_and(thita>=-1/8*np.pi,thita<np.pi/8 ))
    Ta[B0]=1
    B1=np.where(np.logical_or(thita>=7/8*np.pi,thita<-7/8*np.pi))
    Ta[B1]=-1

    B0=np.where(np.logical_and(thita>=np.pi/8,thita<3/8*np.pi))
    Ta[B0]=2
    B1=np.where(np.logical_and(thita>=-7/8*np.pi,thita<-5/8*np.pi))
    Ta[B1]=-2

    B0=np.where(np.logical_and(thita>= 3/8*np.pi,thita<5/8*np.pi ))
    Ta[B0]=3
    B1=np.where(np.logical_and(thita>=-5/8*np.pi,thita<-3/8*np.pi))
    Ta[B1]=-3

    B0=np.where(np.logical_and(thita>= 5/8*np.pi,thita < 7/8*np.pi))
    Ta[B0]=4
    B1=np.where(np.logical_and(thita>= -3/8*np.pi,thita< -1/8*np.pi ))
    Ta[B1]=-4

    A=np.where(Eimg<1)
    Ta[A[0],A[1]]=0
    return Eimg,Ta

def RectangleDbscan(Pon,Eps=2, Min_samples=2):
    Rect=[]
    Min_samples=np.sqrt(2)+0.001
    db = DBSCAN(Eps, Min_samples).fit(Pon)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        class_member_mask = (labels == k)
        xy=Pon[class_member_mask & core_samples_mask]
        if xy.size>0:
            Rect.append([np.min(xy[:,0]),np.min(xy[:,1]),np.max(xy[:,0]),np.max(xy[:,1])])
    return Rect
def heightCluster(Pon,R=15,D=4):
    K=[0,2]
    Clusts=np.array([],dtype=np.int).reshape(0,4)
    poni=Pon[0]
    PointsToNext=0
    Pon=np.delete(Pon, 0, 0)
    while Pon.shape[0]>0:
        A=np.sqrt((poni[K[0]]-Pon[:,K[0]])**2+(poni[K[1]]-Pon[:,K[1]])**2)<=R
        if np.count_nonzero(A)>=D:
            PointsToNext=Pon[A]
            Clusts=np.concatenate((Clusts,poni.reshape(1,4),PointsToNext))
            A=np.where(A==True)
            Pon=np.delete(Pon,A,0)

            while PointsToNext.shape[0] > 0:
                Pi=PointsToNext[0]
                PointsToNext=np.delete(PointsToNext,0,0)
                A=np.sqrt((Pi[0]-Pon[:,K[0]])**2+(Pi[1]-Pon[:,K[1]])**2)<=R

                if np.count_nonzero(A)>=D:
                    A=np.where(A==True)
                    Np=Pon[A]
                    PointsToNext=np.concatenate((PointsToNext,Np))
                    Pon=np.delete(Pon,A,0)
                    Clusts=np.concatenate((Clusts,Np))
        else:
            poni=Pon[0]
            Pon=np.delete(Pon, 0, 0)
    return Clusts
def AreaCluster(Rect,maxE=0.1,minh=0.1):
    Rect=list(Rect)
    Rect2=[]
    E=img.size*maxE

    h=img.shape[1]*minh
    for i in range(len(Rect)):
        top=Rect[i][0]
        left=Rect[i][1]
        bot=Rect[i][2]
        right=Rect[i][3]

        if (bot-top)*(right-left)<E and (bot-top)*(right-left)>0.05*E and (bot-top)>h:
            j=0
            Bool=True
            while Bool and j<len(Rect2):
                if (left-Rect2[j][1])*(right-Rect2[j][3])<=0 and (top-Rect2[j][0])*(bot-Rect2[j][2])<=0:
                    Bool=False
                    Rect2[j][0]=min(top,Rect2[j][0])
                    Rect2[j][1]=min(left,Rect2[j][1])
                    Rect2[j][2]=max(Rect2[j][2],bot)
                    Rect2[j][3]=max(Rect2[j][3],right)
                j+=1
            if Bool:
                Rect2.append([top,left,bot,right])
    Rect=Rect2
    return np.array(Rect).reshape(len(Rect),4)
def Color(ImP,Rect,c):
    ImP=np.array(ImP)
    n_clusters_=Rect.shape[0]
    for I in range(0,n_clusters_):
        pos0x=Rect[I][0]
        pos0y=Rect[I][1]
        pos1x=Rect[I][2]
        pos1y=Rect[I][3]
        ImP[pos0x,pos0y:pos1y]=c
        ImP[pos0x:pos1x,pos0y]=c
        ImP[pos1x,pos0y:pos1y]=c
        ImP[pos0x:pos1x,pos1y]=c

    plt.imshow(ImP)

cutx=1
cuty=1

SetNames=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I',
        'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

Dir="36ClassSet"

scaleIM=28
Imd=1

Net = NNtool(Dir,False)
Net.SetSession()
Net.Initialize_Vars()
gap=2

HEinput=[70,170,170,255,5,50]
for ip in range(1,19):
    # filename="plates2/plate-"+str(ip)+".jpg" original
    filename="PLATES/plate-"+str(ip)
    img=mpimg.imread('../'+filename)
    img=img[cutx:-cutx,cuty:-cuty]
    Per=(383*900/img.size)**(1/2)
    img=spm.imresize(img,(int(img.shape[0]*Per),int(img.shape[1]*Per)),'bicubic')
    IMG=np.array(img)
    if(len(img.shape)==3):
        if(np.max(img)>1):
            img=img
        img=img[:,:,0]*1/3 + img[:,:,1]*1/3 + img[:,:,2]*1/3
    img=img.astype(int)
    img=img.astype(int)


    img=HE(img,
                HEinput[0],
                HEinput[1],
                HEinput[2],
                HEinput[3],
                HEinput[4],
                HEinput[5]
            )
    img=img/255
    Totaltime=0
    ti=time.time()
    Eimg,thita=Cany_skimage(img,3,0.1,0.3)
    Pon=GetPoints(Eimg)
    ti=time.time()-ti
    Totaltime+=ti

    #############
    ImPlace=np.zeros((img.shape[0],img.shape[1],3))

    ImPlace[:,:,0]=np.array(img)
    ImPlace[:,:,1]=np.array(img)
    ImPlace[:,:,2]=np.array(img)

    ti=time.time()
    Rect=RectangleDbscan(Pon,2,2)
    Rect=np.array(Rect).reshape(len(Rect),4)


    Rect=heightCluster(Rect)


    Rect=AreaCluster(Rect)
    ti=time.time()-ti
    Totaltime+=ti

    n_clusters_=Rect.shape[0]
    Label=[]
    for I in range(0,n_clusters_):
        pos0x=Rect[I][0]-1
        pos0y=Rect[I][1]-1
        pos1x=Rect[I][2]+1
        pos1y=Rect[I][3]+1
        Label.append([pos0y])
        Rect1=Eimg[pos0x:pos1x,pos0y:pos1y]
        Plot_Pic=IMG[pos0x:pos1x,pos0y:pos1y]
        thita1=thita[pos0x:pos1x,pos0y:pos1y]

        for i in range(Rect1.shape[0]):
            for j in range(Rect1.shape[1]):
                xstep=0
                ystep=0
                if thita1[i,j]!=0:
                    if thita1[i,j]==1:
                        xstep+=-1
                    elif thita1[i,j]==2:
                        xstep+=-1
                        ystep+=-1
                    elif thita1[i,j]==3:
                        ystep+=-1
                    elif thita1[i,j]==4:
                        xstep+=1
                        ystep+=-1
                    elif thita1[i,j]==-1:
                        xstep+=1
                    elif thita1[i,j]==-2:
                        xstep+=1
                        ystep+=1
                    elif thita1[i,j]==-3:
                        ystep+=1
                    elif thita1[i,j]==-4:
                        xstep+=-1
                        ystep+=1
                    ix=i+xstep
                    jy=j+ystep
                    if not(ix+xstep>=thita1.shape[0] or ix+xstep<0 or  jy+ystep<0 or jy+ystep>=thita1.shape[1]):
                        while thita1[ix+xstep,jy+ystep]==0 and thita1[ix,jy+ystep]==0 and thita1[ix+xstep,jy]==0:
                            Rect1[ix,jy]=1
                            ix+=xstep
                            jy+=ystep
                            if ix+xstep==thita1.shape[0] or ix+xstep<0 or  jy+ystep<0 or jy+ystep==thita1.shape[1]:
                                break
        dx=pos1x-pos0x
        dy=pos1y-pos0y
        a=abs(dx-dy)/2
        z1=int(a)
        z0=int(a)+int(a!=int(a))
        maxD=dx
        if(dx>dy):
            Rect1=np.concatenate((np.zeros((dx,z0)),Rect1,np.zeros((dx,z1))),axis=1)
        elif(dx<dy):
            Rect1=np.concatenate((np.zeros((z0,dy)),Rect1,np.zeros((z1,dy))),axis=0)
            maxD=dy
        Rect1=np.concatenate((np.zeros((maxD,gap)),Rect1,np.zeros((maxD,gap))),axis=1)
        Rect1=np.concatenate((np.zeros((gap,maxD+2*gap)),Rect1,np.zeros((gap,maxD+2*gap))),axis=0)
        ScalePic=ScaleArray(Rect1,scaleIM-2*gap)
        ScalePic=np.concatenate((np.zeros((scaleIM-2*gap,gap)),ScalePic,np.zeros((scaleIM-2*gap,gap))),axis=1)
        ScalePic=np.concatenate((np.zeros((gap,scaleIM)),ScalePic,np.zeros((gap,scaleIM))),axis=0)
        Score  = Net.Layers[-1].eval(feed_dict = {
                 Net.Layers[0]:ScalePic.reshape(1,scaleIM,scaleIM,Imd),
                 Net.keep_prob :np.ones((Net.DroupoutsProbabilitys.shape[0]))
        })
        maxScore=np.max(Score)
        Label[-1].append(str(SetNames[np.where(Score.reshape(36)==maxScore)[0][0]]))
    plt.subplot(7,3,ip)
    plt.axis('off')
    L=''
    for l in sorted(Label, key=lambda Labeli: Labeli[0]):
        L+=l[1]
    plt.title(L)
    plt.imshow(IMG)
plt.subplots_adjust(0,0,1,.95,.5,.41)
plt.show()
