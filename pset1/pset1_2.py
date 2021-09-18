import numpy as np
from scipy.misc import derivative


eps =10**-16

def ndiff(fun,x,full=False): 
       
    dx = 10**(-2) #initial guess
    
    def d1(x):
        #first deriv
        return (fun(x+dx)-fun(x-dx))/(2*dx)
    def d2(x):
        #second deriv
        return (fun(x+dx)+fun(x-dx)-2*fun(x))/dx**2
        
    def d3(x):
        #third deriv
        return (fun(x+dx)+3*fun(x-dx)-3*fun(x)-fun(x-2*dx))/dx**3
        
    dx = ((np.abs(eps*fun(x)))/np.abs(d3(x)))**(1/3) #optimize dx as seen in class

    total_err = (eps*fun(x)/dx)+(d3(x)*dx**2) #roundoff error + theoritical error
    
    if full==True:
        return d1(x), dx, total_err
        
    
    return d1(x)


x = 0.78
f = lambda x : np.arctan(x)
df = lambda x : 1/(1+x**2)

dx= ndiff(f,x,True)[1]

#testing
print('true : ',df(x))
print('deriv : ',ndiff(f,x,False))
print('scipy :', derivative(f,x,dx))
print('\n')
print('diff true/deriv is :',df(x)-ndiff(f,x,False))
print('diff scipy/deriv is :',derivative(f,x,dx)-ndiff(f,x,False))
print('diff true/scipy is :',derivative(f,x,dx)-df(x))