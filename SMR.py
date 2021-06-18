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
###############
    def forward(self, shn, shk, n):
        x = self.f(x)   # x==>(64*64*256)
        shk2 = self.compare_sh(shn, shk, n)  
        x1 = x*shk2  # 
        x = x1 + x   # 
        x = self.max_pooling(x)  
        return x
    def compare_sh(self, shn, shk, n)        
        shn = shn.numpy()          
        shk = shk.numpy()
        shk_normal=shk
        map1 = np.zeros(shk.shape)
        map2 = np.zeros(shk.shape)
        
        for x in range (n-2):
            mean_shn = np.mean(shn[x])
            for i in range shk.shape[0]:
                for j in range shk.shape[1]:
                    if shn[x][i][j] >= shk_normal[i][j]:
                        map1[i][j] = shn[x][i][j]
                    else:
                        map1[i][j] = 0
            
            for i in range shk.shape[0]:
                for j in range shk.shape[1]:
                    if shn[x][i][j] >= mean_shn:
                        map2[i][j] = shn[x][i][j]
                    else:
                        map2[i][j] = 0
            map3 = np.minimum(map1,map2)
            
            
            for i in range shk.shape[0]:
                for j in range shk.shape[1]:
                    if shk[i][j]>=map3[i][j]:   #
                        shk[i][j]=shk[i][j]-map3[i][j]
                    else:
                        shk[i][j]=0
        shk2 = torch.from_numpy(shk) # 
        return shk2
