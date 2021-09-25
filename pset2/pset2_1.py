import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate


#do adaptive integration with legendre poly where closer to the singularity decrease dx

#true answer : z>R -->1/z**2
#              z<R -->0

R = 5
n = 50
zs = np.linspace(-10,10,50)
x = np.linspace(-1,1,n+1) # range of integration

def fun(x):
    y =(R**2)*(z-R*x)/((R**2 + z**2 -2*R*z*x)**(3/2))
    return y

def get_legendre_weights(n):
    #y=Pc - we want to pick c so that Pc goes through y
    #c = P^-1 y (if P is invertible - which it is!)
    #because we oonly care about c_0, then we only need the first
    #fow of P^-1
    x=np.linspace(-1,1,n+1)
    P=np.polynomial.legendre.legvander(x,n)
    Pinv=np.linalg.inv(P)
    coeffs=Pinv[0,:]
    #coeffs=coeffs/coeffs.sum()*n
    return coeffs*n

coeffs=get_legendre_weights(n)

dx=x[1]-x[0]

e_field = []

for z in zs:
    y =(R**2)*(z-R*x)/((R**2 + z**2 -2*R*z*x)**(3/2))
    e_field.append(y)
    my_int=np.sum(coeffs*y)*dx
    int_quad=integrate.quad(fun,-1,1)
    
plt.plot(zs, e_field,'.')
plt.show()
