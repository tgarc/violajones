import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
from itertools import chain

frameSize = 24

features = map(np.array,([[1,0]], [[1],[0]], [[1,0,1]], [[1],[0],[1]], [[1,0],[0,1]]))

fig,ax = plt.subplots()
arr = np.zeros((frameSize,frameSize), dtype=np.uint8)
arr[:] = 127
imdisp = ax.matshow(arr
                    ,interpolation='none'
                    ,cmap=plt.cm.gray
                    ,extent=[0,frameSize,frameSize,0]
                    ,vmin=0,vmax=255)

def init():
    # doesn't work with blitting
    ax.set_xticks(np.arange(frameSize),minor=True)
    ax.set_yticks(np.arange(frameSize),minor=True)
    ax.grid(ls='solid',which='both')

    return imdisp,

def iterkern(kern,frameSize):
    majaxis = np.argmax(kern.shape)
    minaxis = (majaxis+1)%2

    if kern.shape[0] == kern.shape[1]:
        for i in np.arange(1,frameSize//kern.shape[minaxis]+1):
            yield np.kron(kern,np.ones((i,i),dtype=np.uint8))
    else:
        repaxis = [1,1]
        for i in np.arange(1,frameSize//kern.shape[minaxis]+1):
            repaxis[minaxis] = i
            tmpkern = np.tile(kern,repaxis)
            for j in np.arange(1,frameSize//kern.shape[majaxis]+1):
                yield tmpkern.repeat(j,axis=majaxis)

def animate(i,kerniter):
    kern = kerniter.next()    
    arr[:] = 127

    arr[:kern.shape[0],:kern.shape[1]] = kern*255
    imdisp.set_array(arr)

    return imdisp,

kerniter = (k for f in features for k in iterkern(f,frameSize))
anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(kerniter,)
                               , interval=argv[1] if len(argv)>1 else 200
                               , blit=True, repeat=False)
try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close()
