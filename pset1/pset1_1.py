import numpy as np

x=42

#compute deriv for exp(x)
eps=10**-7
f1 = lambda x : np.exp(x)
dx = (eps*f1(x)*10080/(f1(x)*839))**1/3
print(dx)
deriv = (f1(x+dx)-f1(x-dx))/(2*dx)
print(deriv)
print(f1(x),'\n')

#compute deriv for exp(x/100)
eps=10**-16
f1 = lambda x : np.exp(x/100)
third_deriv  = lambda x : (1e10**-6) * np.exp(x/100)
dx = (eps*f1(x)*10080/(third_deriv(x)*839))**1/3
print(dx)
deriv = (f1(x+dx)-f1(x-dx))/(2*dx)
print(deriv)
print(f1(x))