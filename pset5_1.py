import numpy as np
from matplotlib import pyplot as plt

#need to chekc if data evenly spaced in t
#shift x 



def shifter(array,shift):
    '''
    1. take fourier transform of array
    2. get fourier transform of delta fcn with shift
    3. multiply to get fourier transform of convolution
    4. inverse convolution to get shifted array
    '''
    print(len(array))
    pad = shift
    array = np.concatenate((array,np.zeros(pad)))
    N = len(array)
    
    yft = np.fft.fft(array) #step 1
    
    #step 2
    delta = np.zeros(N,dtype='complex')    
    for k in range(N):
        delta[k]= np.exp(-2*np.pi*1J*k*shift/N)
    
    
    mult = yft*delta #step 3
    
    yshifted = np.fft.ifft(mult)

    

    return yshifted[:N-shift]


def gaussian(x,mu=0):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-0.5*(x-mu)**2)

x = np.linspace(-10,10,500)
#plt.plot(X,gaussian(X))
#plt.plot(X,gaussian(X,5))
#plt.show()
 
y = gaussian(x)
'''
y = np.arange(0,10)
x = np.arange(len(y))'''
a = 5

shifted = shifter(y,a)
y_shift = np.abs(shifted)


plt.plot(x, y, color='b')

plt.plot(x,y_shift)

#plt.plot(np.arange(len(shifted))-(len(shifted)/2),np.real(shifted))
#plt.xlim(-10,10)


plt.show()