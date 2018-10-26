### HW4
import numpy as np
import matplotlib.pyplot as plt

N = np.arange(10000)
d = 0.05
d_vc = 50

mHofN = N
eps = 0.05

mH = N*2
vc = np.sqrt((8/N)*np.log(4*mH/d))
mH = N
radPen = np.sqrt(2*np.log(2*N*mH)/N) + np.sqrt((2/N)*np.log(1/d)) + (1/N)
mH = N*2
parVanBro = np.sqrt((1/N)*(2*eps+np.log(6*mH/d)))
mH = np.square(N)
devroye = np.sqrt((1/(2*N))*(4*eps*(1+eps) + np.log(4*mH/d)))

plt.plot(vc)
plt.plot(radPen)
plt.plot(parVanBro)
plt.plot(devroye)


