import sys
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt

def trainPerceptron(X,Y,w):
    H = h(X,w)
        
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
    
    w = np.add(w,yn*xn) ### train weights!
    
    percentCorrect = (NUM_PTS-numWrongPts) / NUM_PTS
    return w, percentCorrect

def getRandLineInR2(lowerLim, upperLim):
    targetPts = np.random.uniform(low=lowerLim, high=upperLim, size=(2,2))
    xTargets = np.array(targetPts[:,0])
    yTargets = np.array(targetPts[:,1])
    A = np.vstack([xTargets, np.ones(len(xTargets))]).T
    mTarget, cTarget = np.linalg.lstsq(A, yTargets, rcond=None)[0]

    ### sanity check / debugging    
#    plt.plot(xTargets, yTargets, 'o', label='random points', markersize=5)
#    xs = np.linspace(-1,1,num=100)
#    plt.plot(xs, mTarget*xs + cTarget, 'r', label='random hypothesis')
    
    return mTarget, cTarget
    
def getY(NUM_PTS, mTarget, cTarget, X):
    Y = np.full((NUM_PTS,),np.nan) 
    for ptInd in range(0,NUM_PTS):
        threshold = np.dot(mTarget,X[2,ptInd]) + cTarget
        if X[2,ptInd] > threshold:
            Y[ptInd] = 1
        elif X[2,ptInd] < threshold:
            Y[ptInd] = -1
    return Y

def signActivation(X,w):
#    sanityCheck = W[-1] * x[-1]
#    print(sanityCheck)
    return np.sign(np.dot(w.T, X))

### version for regression
#def getEin(X,Y,w):
##    X = X.T
#    sumSqrdError = 0
#    for ptInd in range(0,NUM_PTS):
#        sumSqrdError += np.square(np.dot(X[ptInd,:],w)-Y[ptInd])
#    Ein = sumSqrdError / NUM_PTS
#    return Ein
    
def getEin(h,Y):    
    numCorrect = 0
    for ptInd in range(0,len(h)):
        if h[ptInd] == Y[ptInd]:
            numCorrect += 1
    Ein = 1.0 - (numCorrect/len(h))
    return Ein

def runLinearRegression(X,y):
    w = np.full((3,),float(0)) ### w0 is for the bias
    X = X.T
#    Xdag = np.dot(inv(np.dot(X.T,X)),X.T)
    Xdag = pinv(X)
    w = np.dot(Xdag,y)
    X = X.T
    h = signActivation(X,w)
    
    Ein = getEin(h,y)
    return w, Ein

### USER PARAMETERS
NUM_RUNS = 100000 
NUM_PTS = 100
#CONVERGENCE_THRESH = 0.999

Xin = np.random.uniform(low=-1,high=1, size=(3,NUM_PTS))
Xin[0,:] = 1 

####### plot input data 
#X = Xin
#plt.figure()
#plt.plot(X[1,:], X[2,:], 'o')
### equivalent
#plt.scatter(X[:,0], X[:,1])

numIterationsPerRun = []
fBySlopeNintercept = []
Gs = []
Eins = []
for runInd in range(0,NUM_RUNS):
    
    Xout = np.random.uniform(low=-1,high=1, size=(3,NUM_PTS))
    Xout[0,:] = 1
    
    ### correct output defined
    mTarget, cTarget = getRandLineInR2(-1,1)
    fBySlopeNintercept.append([mTarget, cTarget])
    y = getY(NUM_PTS, mTarget, cTarget, X)
    
    w, Ein = runLinearRegression(X,y)
    
    ### sanity check plotting
#    xs = np.linspace(-1,1,num=100)
#    plt.plot(xs, w[1]*xs + w[0], 'g', label='linear regression')
#    plt.xlim(-1,1)
#    plt.ylim(-1,1)
#    plt.legend()
    
    ### saving results for later
    print('Run: {2}; Ein: {1}\nw: {0};\n'.format(w, Ein, runInd))    
    Gs.append(w)
    Eins.append(Ein)
    
    
    
avgEin = np.mean(Ein)
print('avg Ein: {0}'.format(avgEin))

#sys.exit('breaking loop via sys.exit')
    

    

#    
#           
#    converged = False
#    iterationNum = 1
#    while not converged:
#        
##        w = trainPerceptron(X,Y)
#        
#        print('Run num: {2}; Iteration num: {0}; Ein: {1}\n'.format(iterationNum, Ein, runInd))
#        
#        if percentCorrect > CONVERGENCE_THRESH:
#            converged = True 
#        iterationNum += 1
#    numIterationsPerRun.append(iterationNum)
#
#    
#avgNumIterations = np.mean(numIterationsPerRun)
#print('{0} iterations required on avg to reach convergence'.format(avgNumIterations))
#print('num pts: {0}; accuracy for convergence: {1}'.format(NUM_PTS,CONVERGENCE_THRESH))
        
        