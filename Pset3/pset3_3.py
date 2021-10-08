import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('dish_zenith.txt')

x = data[:,0]
y = data[:,1]
z = data[:,2]

#construct A matrix
nd = len(x)
nm = 4


def param(N):
    '''
    input : N matrix which has var as diagonal
    output :
    m-->parameters for lin least square  fit minimizing chi-square
    y_pred-->the value predicted for z from A and m
    stud_res-->studentized residuals
    chi_square-->chi_square
    '''
    A = np.zeros([nd,nm])
    A[:,0] = x**2 + y**2
    A[:,1] = -2*x
    A[:,2] = -2*y
    A[:,3] = 1
    
    lhs = A.T@np.linalg.inv(N)@A
    rhs = A.T@np.linalg.inv(N)@z 
    m = np.linalg.inv(lhs)@rhs
    
    y_pred = A@m
    res = (z - A@m)
    chi_square = res.T@np.linalg.inv(N)@res
    cov_mat = np.linalg.inv(lhs)
    error_m = np.sqrt(np.diag(cov_mat))
    error_a = error_m[0]
    
    return m, y_pred, chi_square/(len(z)-4), error_a


N = np.eye(len(z))
m, y_pred,  chi_square, error_a = param(N)
print('Fit parameters (a, x_0, y_0, z_0) are :',m[0],m[1]/m[0],m[2]/m[0],m[3]-(m[1]**2/m[0])-(m[2]**2/m[0]))
print('The reduced chi-square for the fit is with N=I:',chi_square)

noise = (z-y_pred) # residuals

#hence, take that error in z is proportional to absolute value of noise
# get that std dev is np.sqrt(np.abs(noise))

#assume uncorrelated noise
N = np.sqrt(noise*noise*np.eye(len(z))) #value of variance is np.sqrt(noise*noise)
#unc_data = np.sqrt(np.diag(N))
m, y_pred, chi_square, error_a = param(N)

print('The value for parameter a is :',m[0],'+/-',error_a)
print('The reduced chi-square for the fit using the new N is :',chi_square)


#focal length : f=1/4a where a has unit of 1/length, the error on f is computed by propagating the error
f= 1/(4*m[0]*1000) #compute focal length
print('Focal length :',f,'+/-',(error_a/m[0])*f)


#fig, (ax1,ax2) = plt.subplots(2,1)
#ax1.plot(x, noise/unc_data,'.')
#ax2.plot(y, noise/unc_data,'.')

#ax1.errorbar(x,z,yerr=np.sqrt(np.diag(N)),fmt='.')

#plt.show()
