import numpy as np
import random

nEXPERIMENTS = 100000 
nCOINS = 1000
nFLIPS_PER_COIN = 10

v1 = []
vMin = []
vRand = []
for expIdx in range(1,nEXPERIMENTS+1):
    
    exp = np.random.choice([0,1],size=(nCOINS,nFLIPS_PER_COIN)) ### 1 = heads; 0 = tails
    counts = exp.sum(1)
    c1 = exp[0,:]
    cMin = exp[counts.argmin(),:]
    randIdx = random.randint(0,exp.shape[0]-1) 
    cRand = exp[randIdx,:]
    
    v1.append(np.count_nonzero(c1 == 1)/np.size(c1))
    vMin.append(np.count_nonzero(cMin == 1)/np.size(cMin))
    vRand.append(np.count_nonzero(cRand == 1)/np.size(cRand))
    
    print('Running experiment: {0}; selecting random idx: {1}'.format(expIdx,randIdx))

mean_v1 = np.mean(v1)
mean_vMin = np.mean(vMin)
mean_vRand = np.mean(vRand)

print('\nMeans:\nv1: {0}\nvMin: {1}\nvRand: {2}'.format(mean_v1,mean_vMin,mean_vRand))
    

    