import numpy as np
import matplotlib.pyplot as plt

'''
Cant use gaussian as bounding function as it goes to 0 faster than an exp even for high sigma

'''

b=1

#exponential distribution
def exp(x):
    return b*np.exp(-b*x)

def randl(n):
    q=np.pi*(np.random.rand(n)-0.5)
    return np.tan(q)

# Lorentzian Distribution
def l_pdf(x,gamma=1):
    return 1/(gamma*np.pi) * 1/(1 + np.power(x/gamma,2))

n = 1000000

# Generate random loretnzian and uniform numbers
ys = randl(n) #horizontal number

#depending on b, scale the lorenztian such that it is greater than exp
if b>=1 : 
    M = np.pi*b 
elif b<1:
    M=np.pi*5
    
keep1 = ys>=0 #if ys<0 then dont want
ys_ = ys[keep1]
us = np.random.rand(len(ys_))

# exp and Lorentzian probabilities of the random numbers
fs = exp(ys_) #exp
gs = l_pdf(ys_) #lorentzian

# Rejection Step
keep = us *(M * gs)< fs 
exp_rands = ys_[keep]

xs = np.linspace(0,10,10000)
e = exp(xs)
l = M*l_pdf(xs)

# Plotting Results
plt.figure()
plt.hist(t,bins=50,density=True, label="Samples")
xs = np.linspace(0,max(t),1000)
plt.plot(xs,b*np.exp(-b*xs),label="Expected")
plt.legend()
plt.xlabel(f"Method is {len(t)/(2*N) * 100}% Efficient")
plt.savefig('q2_lorentzian.png')
plt.show()

