import numpy as np
import torch
import torch.nn as nn

class SMR(nn.Module):
    def __init__(self):
        super(SMR, self).__init__()
##################
        self.f = featur_layer()
        self.max_pooling = nn.MaxPool2d(self.input_channel, stride=1)
        self.map3 = self.compare_sh()

################
    def forward(self, x, shn, shk):
        x = self.f(x)   # 
        map3 = self.compare_sh(shn, shk)  # 
        x1 = x*map3  # 
        x = x1 + x   # 
        x = self.max_pooling(x)  # 
        
        return x
    def compare_sh(self, shn, shk):       
        shn = shn.numpy()  #
        shk = shk.numpy()
        max_sh = np.maximum(shn, shk)
        min_sh = np.minimum(shn, shk)
        mean_shn = np.mean(shn)
        map1 = np.zeros(shn.shape)
        for i in range shn[0]:
            for j in range shn[1]:
                if shn[i][j] >= shk[i][j]:
                    map1[i][j] = shn[i][j]
                else:
                    map1[i][j] = 0
        map2 = np.zeros(shn.shape)
        for i in range shn[0]:
            for j in range shn[1]:
                if shn[i][j] >= mean_shn:
                    map2[i][j] = shn[i][j]
                else:
                    map2[i][j] = 0
        map3 = shnk - np.minmum(map1,map2)
        map3 = torch.from_numpy(map3) # 
        return map3



