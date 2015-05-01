import numpy as np
import itertools as it

features = map(lambda x: np.array(x,dtype=np.int32)
               ,([[-1,1]], [[-1],[1]], [[-1,1,-1]], [[-1],[1],[-1]], [[-1,1],[1,-1]]))

def iterkern(kern,frameSize):
    if kern.shape[0] == kern.shape[1]:
        for i in np.arange(1,frameSize//2+1):
            yield np.kron(kern,np.ones((i,i),dtype=np.uint8))
    else:
        majaxis = np.argmax(kern.shape)
        minaxis = (majaxis+1)%2
        repaxis = [1,1]
        for i in np.arange(1,frameSize//kern.shape[minaxis]+1):
            repaxis[minaxis] = i
            tmpkern = np.tile(kern,repaxis)
            for j in np.arange(1,frameSize//kern.shape[majaxis]+1):
                yield tmpkern.repeat(j,axis=majaxis)


def rectsum(intimg,rect):
    """
    Expects integral image to have a zero padding before each axis
    """
    x,y,w,h = rect
    return intimg[y,x]+intimg[y+h,x+w]-intimg[y,x+w]-intimg[y+h,x]


def extractfeature(feat,img,frameSize):
    # it's not entirely impossible that overflow won't occur here, but we can
    # safely assume that training images will not be large enough to cause
    # overflow
    intimg = img.astype(np.int32)
    intimg.cumsum(1,out=intimg).cumsum(0,out=intimg)
    intimg = np.pad(intimg,(1,0),mode='constant',constant_values=0)
        
    nsubblocks = feat.size
    majaxis = np.argmax(feat.shape[::-1])

    f= feat.flatten()
    for kern in iterkern(feat,frameSize):
        h,w = kern.shape
        boxsize = [w,h]
        if h == w:
            boxsize = [w//2,h//2]
        else:
            boxsize[majaxis] //= nsubblocks

        diff = np.zeros((img.shape[0]-h+1,img.shape[1]-w+1),dtype=np.int32)
        for x,y in it.product(np.arange(frameSize-w+1),np.arange(frameSize-h+1)):
            a = [x,y]
            if feat.shape[0] == feat.shape[1]:
                b = [x+w//2,y]
                c = [x,y+h//2]
                d = [x+w//2, y+h//2]
                diff[y,x] = f[0]*rectsum(intimg,a+boxsize) + f[1]*rectsum(intimg,d+boxsize) \
                            + f[2]*rectsum(intimg,b+boxsize) + f[3]*rectsum(intimg,c+boxsize)
            elif feat.shape[majaxis] == 3:
                b = list(a)
                b[majaxis] += boxsize[majaxis]//nsubblocks
                c = list(a)
                c[majaxis] += 2*boxsize[majaxis]//nsubblocks
                diff[y,x] = f[0]*rectsum(intimg,a+boxsize) + f[1]*rectsum(intimg,c+boxsize) \
                            + f[2]*rectsum(intimg,b+boxsize)
            else: # elif f.shape[majaxis] == 2:
                b = list(a)
                b[majaxis] += boxsize[majaxis]//nsubblocks
                diff[y,x] = f[0]*rectsum(intimg,a+boxsize) + f[1]*rectsum(intimg,b+boxsize)
        yield kern,diff
                
