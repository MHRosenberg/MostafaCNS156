import numpy as np
import random
nEXPERIMENTS = 10
nCOINS = 4
nFLIPS_PER_COIN = 10


exp = np.random.choice([0,1],size=(nCOINS,nEXPERIMENTS))

sums = exp.sum(1)

test = random.randint(0,exp.shape[0])