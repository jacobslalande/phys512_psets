import numpy as np
from matplotlib import pyplot as plt


def conv_safe(f,g):
    m = len(f)
    n = len(g)
    l = m
    #padding of smaller length array
    if m != n:
        if m > n :
            l = m
            diff = m-n
            g = np.concatenate((g,np.zeros(diff)))
        elif n > m :
            l = n
            diff = n-m
            f = np.concatenate(f,np.zeros(diff))
    padding = int(l/2)
    
    f = np.concatenate((np.zeros(padding),f))
    g = np.concatenate((np.zeros(padding),g))

    
     
    h=np.fft.irfft(np.fft.rfft(f)*np.fft.rfft(g))
    
    return h

def gaussian(x,mu=0):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-0.5*(x-mu)**2)
    
    
x = np.linspace(-10,10,499)
y = gaussian(x)


con = conv_safe(y,x)
print('the length of the output array is the length of the longest input array +0.5(length of longest input array)',len(con))

plt.plot(con)
plt.show()