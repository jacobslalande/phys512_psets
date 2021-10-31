import numpy as np
from matplotlib import pyplot as plt


def corr(f,g):
    ft_f = np.fft.fft(f)
    conj_ft_g = np.conjugate(np.fft.fft(g))
    
    return np.fft.ifft(ft_f*conj_ft_g)

def shifter(array,shift):
    '''
    1. take fourier transform of array
    2. get fourier transform of delta fcn with shift
    3. multiply to get fourier transform of convolution
    4. inverse convolution to get shifted array
    '''
    shift = int(shift)


    N = len(array)
    
    yft = np.fft.fft(array) #step 1
    
    #step 2
    delta = np.zeros(N,dtype='complex')    
    for k in range(N):
        delta[k]= np.exp(-2*np.pi*1J*k*shift/N)
    
    mult = yft*delta #step 3
    
    yshifted = np.fft.ifft(mult) #step 4
    
    return yshifted


def shift_corr(array,shift):

    shifted_arr = shifter(array,shift) #shift array
    c = corr(array,shifted_arr) #correlation with shifted array and array

    return np.fft.fftshift(c)
    

def gaussian(x,mu=0):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-0.5*(x-mu)**2)

x = np.linspace(-10,10,500)
y = gaussian(x)


cs = shift_corr(y,150)
plt.plot(cs,label='cs')
plt.legend()

cs = shift_corr(y,0)
plt.plot(cs)

plt.show()