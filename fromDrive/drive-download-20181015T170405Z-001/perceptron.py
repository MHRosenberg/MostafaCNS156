import numpy as np
import matplotlib.pyplot as plt

### USER PARAMETERS
NUM_RUNS = 1000 ############################## 1000
NUM_PTS = 100
CONVERGENCE_THRESH = 0.999

def h(x):
#    sanityCheck = W[-1] * x[-1]
#    print(sanityCheck)
    return np.sign(np.dot(W.T, x))
plt.close('all')

X = np.random.uniform(low=-1,high=1, size=(3,NUM_PTS))
X[0,:] = 1 

####### plot input data 
plt.plot(X[:,0], X[:,1], 'o')
### equivalent
#plt.scatter(X[:,0], X[:,1])

numIterationsPerRun = []
for runInd in range(0,NUM_RUNS):

    targetPts = np.random.uniform(low=-1, high=1, size=(2,2))
    
    
    xTargets = np.array(targetPts[:,0])
    yTargets = np.array(targetPts[:,1])
    A = np.vstack([xTargets, np.ones(len(xTargets))]).T
    
    mTarget, cTarget = np.linalg.lstsq(A, yTargets, rcond=None)[0]
    
#    plt.plot(xTargets, yTargets, 'o', label='random points', markersize=5)
#    xs = np.linspace(-1,1,num=100)
#    plt.plot(xs, mTarget*xs + cTarget, 'r', label='random hypothesis')
#    plt.legend()
    #plt.show()
       
    
    ### correct output defined
    Y = np.full((NUM_PTS,),np.nan) 
    for ptInd in range(0,NUM_PTS):
        threshold = np.dot(mTarget,X[2,ptInd]) + cTarget
        if X[2,ptInd] > threshold:
            Y[ptInd] = 1
        elif X[2,ptInd] < threshold:
            Y[ptInd] = -1
        
    converged = False
    iterationNum = 1
    W = np.full((3,),float(0)) ### w0 is for the bias
    while not converged:
        
        H = h(X)
        
        ### get list of misclassified pts
        wrongInds = []
        for ptInd in range(1,len(H)):
            if H[ptInd] != Y[ptInd]:
                wrongInds.append(ptInd)
        numWrongPts = len(wrongInds)
        
        ### choose random index
        if wrongInds != []:
            randWrongPtInd = np.random.choice(wrongInds)
        
        ### select random misclassified pt
        xn = X[:,randWrongPtInd]
        yn = Y[randWrongPtInd]
        
        W = np.add(W,yn*xn)
        
        percentCorrect = (NUM_PTS-numWrongPts) / NUM_PTS
        
        if percentCorrect > CONVERGENCE_THRESH:
            converged = True 
        
        print('Run num: {2}; Iteration num: {0}; Percent correct: {1}'.format(iterationNum, percentCorrect, runInd))
#        print('Y: {0}'.format(Y))
#        print('H: {0}'.format(H))
#        print('W: {0}'.format(W))
        
        print('\n')
        iterationNum += 1
    numIterationsPerRun.append(iterationNum)
    
avgNumIterations = np.mean(numIterationsPerRun)

print('{0} iterations required on avg to reach convergence'.format(avgNumIterations))
print('num pts: {0}; accuracy for convergence: {1}'.format(NUM_PTS,CONVERGENCE_THRESH))
        
        