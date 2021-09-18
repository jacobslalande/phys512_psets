import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

'''
take arbitrary voltage and interpolate to return a temp
do estimate of the error in interpolation (use difference data with

**also probably need to give interpolation for last points
'''

dat = np.loadtxt('lakeshore.txt')


def lakeshore(V,data):
    '''
    V is either a number or an array)
    output : interpolated temp and uncertainty on temp
    '''
    temp = data[:,0]
    volt= data[:,1]
    deriv = data[:,2]

    
    #reverse order of data
    x = volt[::-1]
    y = temp[::-1]
    dxdy = (deriv[::-1])**-1
    
    #plt.plot(x,y,'.')
    #plt.show()
    xx = np.linspace(x[0],x[-1],20001)
    inter = interpolate.pchip_interpolate(x,y,V) #instead of spline since expect for y to be wtihin bounds of (y1,y2)
    err = interpolate.pchip_interpolate(x,dxdy,V,3)*10**-7/(4*3*2) #epsilon times 4th deriv found using interpolation
    return inter, np.abs(err)

print(lakeshore([1,0.5,8],dat))

    
