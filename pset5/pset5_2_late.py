import numpy as np
from matplotlib import pyplot as plt


def corr(f,g):
    ft_f = np.fft.fft(f)
    conj_ft_g = np.conjugate(np.fft.fft(g))
    
    return np.fft.ifft(ft_f*conj_ft_g)


def gaussian(x,mu=0):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-0.5*(x-mu)**2)
    
    
x = np.linspace(-10,10,500)
y = gaussian(x)

corr = np.fft.fftshift(corr(y,y))

plt.plot(np.abs(corr))
plt.show()
plt.savefig('corr_gaussian.pdf')
