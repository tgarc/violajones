import numpy as np

features = map(np.array,([[1,0]], [[1],[0]], [[1,0,1]], [[1],[0],[1]], [[1,0],[0,1]]))

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
