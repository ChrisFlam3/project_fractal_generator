import numpy as np

#general class for data handling
#change name
class ifs_general(object):
    
    def __init__(self):
        self.boundedFunctionScale=1
        self.affineTransforms=np.zeros((6,8))
        self.current=[0 for i in range(6)]
        self.unlinearNum=0
        self.affineNum=2
        self.zoom=1.0
        self.gauss=9
        self.iterationsInMld=9
        self.renderType=0
        self.postTransforms=[[0 for i in range(6)] for i in range(6)]
        for i in range(6):
            self.postTransforms[i][0] = 1
            self.postTransforms[i][4] = 1


