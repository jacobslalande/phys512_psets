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
    shift = int(shift)
    pad = shift
    array = np.concatenate((array,np.zeros(pad)))
    N = len(array)
    
    yft = np.fft.fft(array) #step 1
    
    #step 2
    delta = np.zeros(N,dtype='complex')    
    for k in range(N):
        delta[k]= np.exp(-2*np.pi*1J*k*shift/N)
    
    mult = yft*delta #step 3
    
    yshifted = np.fft.ifft(mult) #step 4
    #yshifted = yshifted[:N-shift]
    
    return np.abs(yshifted)


def gaussian(x,mu=0):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-0.5*(x-mu)**2)

x = np.linspace(-5,5,500)
y = gaussian(x)

shift = len(x)/2

y_shift = shifter(y,shift)

plt.plot(y)
plt.plot(y_shift)
plt.show()

plt.savefig('gauss_shift.png')