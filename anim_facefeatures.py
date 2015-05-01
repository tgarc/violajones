import h5py
import cv2
import numpy as np
import itertools as it
import features as feat
import matplotlib.pyplot as plt
from sys import argv
import matplotlib.animation as animation


frameSize = 19
arr = np.zeros((frameSize,frameSize), dtype=np.uint8)

fig,(imgax,kax,fimgax) = plt.subplots(1,3)  
kdisp = kax.matshow(arr,interpolation='none',cmap='gray',extent=[0,frameSize,frameSize,0],vmin=0,vmax=255)
fdisp = fimgax.imshow(arr,cmap='gray',interpolation='none',extent=[0,frameSize,frameSize,0],vmin=0,vmax=255)
imdisp = imgax.imshow(arr,cmap='gray',interpolation='none',extent=[0,frameSize,frameSize,0],vmin=0,vmax=255)

def init():
    fimgax.set_xticks([])
    fimgax.set_yticks([])
    kax.set_xticks(np.arange(frameSize),minor=True)
    kax.set_yticks(np.arange(frameSize),minor=True)
    kax.grid(ls='solid',which='both')

    return kdisp,fdisp,imdisp

imgfeatures = iter([])
def animate(i,features,faces):
    global imgfeatures
    update = []

    try:
        kern,thresh = imgfeatures.next()
    except StopIteration:
        img = faces.next()

        imdisp.set_data(img)
        update.append(imdisp)

        imgfeatures = ((kern,thresh) for f in feat.features for kern,thresh in feat.extractfeature(f,img,frameSize))
        kern,thresh = imgfeatures.next()

    kern[kern<0] = 0
    arr[:] = 127
    arr[:kern.shape[0],:kern.shape[1]] = kern*255
    kdisp.set_array(arr)

    normed = cv2.normalize(thresh,thresh,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)*255
    padded = np.ones_like(arr)*255
    padded[:normed.shape[0],:normed.shape[1]] = normed
    fdisp.set_data(padded)
    
    update.extend([kdisp,fdisp])

    return update


faces_fh = h5py.File('faces.hdf5','r')
faces = iter(faces_fh['faces'])
anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(imgfeatures,faces)
                               , interval=argv[1] if len(argv)>1 else 200
                               , blit=True, repeat=False)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close()
