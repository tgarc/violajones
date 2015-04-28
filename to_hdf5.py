import numpy as np
import h5py
import cv2
from glob import glob


with h5py.File("faces.hdf5",'w') as faces_fh:
    facenames = glob('VarianceFaces/*.pgm')
    nonfacenames = glob('VarianceNonFaces/*.pgm')
    data = np.empty((len(facenames)+len(nonfacenames),19,19),dtype=np.uint8)
    classes = np.zeros(len(facenames)+len(nonfacenames),dtype=np.bool)

    i = 0
    for fname in facenames:
        data[i] = cv2.imread(fname,flags=0)
        i += 1
    classes[:i] = True
    for fname in nonfacenames:
        data[i] = cv2.imread(fname,flags=0)
        i += 1

    faces_fh.create_dataset('faces',data=data,chunks=True)
    faces_fh.create_dataset('isface',data=classes,chunks=True)

