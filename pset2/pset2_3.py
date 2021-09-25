import numpy as np
from matplotlib import pyplot as plt

#only need to extend cheb fit to 0, 0.5

#range
a = 0.5
b = 1

x=np.linspace(a,b,100)
y=np.log2(x)

#rescale x range
x_scale = (x-0.5*(b+a))/(0.5*(b-a))

#approx log2 with high number of coefficients
total = 50
coeff = np.polynomial.chebyshev.chebfit(x_scale,y,total)

i = 0
while i < len(coeff):
    #find minimal number of coeff which gives error of approx 10**-6
    truncated_coeff = coeff[:total-i]
    error = np.abs(np.sum(truncated_coeff)) #the error is determine by the sum of the high order coefficients not used in the modeling
    if error > 1e-6:
        truncated_coeff = coeff[:total-i+1]
        break
    i+=1
    
a=0.01
b=0.5
x_ = np.linspace(a,b,5000)
y_true = np.log2(x_)
x_scale2 = (x_-0.5*(b+a))/(0.5*(b-a))


def mylog2(x):
    #take natural log
    assert(x>0)
    mantissa, exp = np.frexp(x)
    e = np.exp(1)
    top = (np.polynomial.chebyshev.chebval(mantissa,truncated_coeff)+exp)
    bot = np.polynomial.chebyshev.chebval(np.frexp(e)[0],truncated_coeff)+np.frexp(e)[1]
    ln = top/bot
    
    return ln

x = 0.5
print('log diff:',mylog2(x)-np.log(x))

