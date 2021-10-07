import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


#compare with fitting with scipy

data = np.loadtxt('dish_zenith.txt')

x = data[:,0]
y = data[:,1]
z = data[:,2]



def func(X, a, x_0, y_0, z_0):
    x, y = X
    z = a*((x-x_0)**2+(y-y_0)**2)
    return z


def fun(X,a,b,c,d):
    x, y = X
    z = (x**2 + y**2)*a - (2*x*b) - (2*x*c) + d
    return z

popt, pcov = curve_fit(fun, (x,y), z)

#print(popt)

nd = len(x)
nm = 4
A = np.zeros([nd,nm])
A[:,0] = x**2 + y**2
A[:,1] = -2*x
A[:,2] = -2*y
A[:,3] = 1

'''
u,s,v=np.linalg.svd(A,0) #always give a 0 so that it is efficient
s = np.eye(4)*s
A_svd = u@s@v.T
param = v@np.linalg.inv(s)@u.T@z
print(param)
y_true = A_svd@param'''

lhs = A.T@A #A transpose times A
rhs = A.T@z # y is d (data) in notes
m = np.linalg.inv(lhs)@rhs
y_pred = A@m

fig, (ax1,ax2, ax3 ,ax4) = plt.subplots(4,1)
ax1.plot(x, y_pred-z,'.')
ax2.plot(y, y_pred-z,'.')

noise = z-y_pred

chi_square = noise.T@noise
print(chi_square)

N = np.eye(len(noise))*noise
print(m)
lhs = A.T@np.linalg.inv(N)@A
rhs = A.T@np.linalg.inv(N)@z
m = np.linalg.inv(lhs)@rhs

print(m)


y_true = A@m


noise = np.array([noise])
noise = noise.T

m_noise = np.linalg.inv(lhs)@A.T@np.linalg.inv(N)@noise


m_noise = np.matrix.flatten(m_noise)
print(m_noise)
m_new = m-m_noise
print(m_new)

y_true = A@m_new

'''
want error such that chi square is 1? seems wrong b/c adjust
likeliness of model according to my needs

-go by section to find error

-plug noise in diag of N and compute m again

-want to flatten noise at extremity of x, y

'''


#fig, (ax1,ax2) = plt.subplots(2,1)

ax3.plot(x, y_true-z,'.')
ax4.plot(y, y_true-z,'.')

plt.show()
