import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import time

half_life=[4.468e9,24.1/365,6.7/8760,245500,75380,1600,3.8235/635,3.1/(60*24*365),26.8/(60*24*365),19.9/(60*24*365),(164.3e-6)/(3600*24*365),22.3,5.015,138.376/365]

def fun(x,y):
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]/half_life[0]
    dydx[1]=-dydx[0]-y[1]*np.log(2)/half_life[1]
    dydx[2]=-dydx[1]-y[2]*np.log(2)/half_life[2]
    dydx[3]=-dydx[2]-y[3]*np.log(2)/half_life[3]
    dydx[4]=-dydx[3]-y[4]*np.log(2)/half_life[4]
    dydx[5]=-dydx[4]-y[5]*np.log(2)/half_life[5]
    dydx[6]=-dydx[5]-y[6]*np.log(2)/half_life[6]
    dydx[7]=-dydx[6]-y[7]*np.log(2)/half_life[7]
    dydx[8]=-dydx[7]-y[8]*np.log(2)/half_life[8]
    dydx[9]=-dydx[8]-y[9]*np.log(2)/half_life[9]
    dydx[10]=-dydx[9]-y[10]*np.log(2)/half_life[10]
    dydx[11]=-dydx[10]-y[11]*np.log(2)/half_life[11]
    dydx[12]=-dydx[11]-y[12]*np.log(2)/half_life[12]
    dydx[13]=-dydx[12]-y[13]*np.log(2)/half_life[13]
    dydx[14]=-dydx[13]
    return dydx



y0=np.zeros(15)
y0[0]=1
x0=0
x1=10
range_t = np.linspace(x0,x1,1000)

ans_stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau',t_eval=range_t)
print(ans_stiff.y)

fig, (ax1,ax2) = plt.subplots(2,1)

ax1.plot(ans_stiff.t,ans_stiff.y[1]/ans_stiff.y[0],'.',label='data for PB206/U238')
ax1.set_ylabel('N')
ax1.legend()

x0=0
x1=1
range_t = np.linspace(x0,x1,1000)
ans_stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau',t_eval=range_t)
ax2.plot(ans_stiff.t,ans_stiff.y[3]/ans_stiff.y[1],'.',label='data for Th230/U234')
ax2.set_ylabel('N')
ax2.set_xlabel('Time (yrs)')
ax2.legend()

plt.show()
