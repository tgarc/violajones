import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
import features as feat

frameSize = 19

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

def animate(i,kerniter):
    kern = kerniter.next()    
    arr[:] = 127
    kern[kern<0] = 0
    arr[:kern.shape[0],:kern.shape[1]] = kern*255
    imdisp.set_array(arr)

    return imdisp,

kerniter = (k for f in feat.features for k in feat.iterkern(f,frameSize))
anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(kerniter,)
                               , interval=argv[1] if len(argv)>1 else 200
                               , blit=False, repeat=False)
try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close()
