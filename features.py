import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sys import argv

frameSize = 24

features = ( (1,2), (2,1), (1,3), (3,1), (2,2) )

fig,ax = plt.subplots()
arr = np.zeros((frameSize,frameSize), dtype=np.uint8)
arr[:] = 63
imdisp = ax.matshow(arr
                    ,interpolation='none'
                    ,cmap=plt.cm.gray
                    ,extent=[0,frameSize,frameSize,0]
                    ,vmin=0,vmax=255)
dims = ((i,j) for j in xrange(1,frameSize+1) for i in xrange(2,frameSize+1,2))

def init():
    # doesn't work with blitting
    ax.set_xticks(np.arange(frameSize),minor=True)
    ax.set_yticks(np.arange(frameSize),minor=True)
    ax.grid(ls='solid',which='both')

    return imdisp,ax

def animate(i):
    h,w = dims.next()
    if h == 2: arr[:] = 63
        
    arr[:h//2,:w] = 0
    arr[h//2:h,:w] = 255
    imdisp.set_array(arr)

    return imdisp,

anim = animation.FuncAnimation(fig, animate, init_func=init
                               , interval=argv[1] if len(argv)>1 else 200
                               , blit=False, repeat=False)
try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close()
