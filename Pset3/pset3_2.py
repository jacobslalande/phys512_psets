import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import time

#half_life = [3.156*4.468e16,24.10*24*3600,6.7*60*60,245500*3.154e7,75380*3.154e7,1600*3.154e7,
             #3.8235*24*3600,3.10*60,26.8*60,19.9*60,164.3e-6,22.3*3.154e7,5.015*3.154e7,138.376*24*3600]
half_life=[4.468e9,24.1/365,6.7/8760,245500,75380,1600,3.8235/635,3.1/(60*24*365),26.8/(60*24*365),19.9/(60*24*365),(164.3e-6)/(3600*24*365),22.3,5.015,138.376/365]

def fun(x,y):
    #let's do a 2-state radioactive decay
    half_life = [3.156*4.468e16,24.10*24*3600,6.7*60*60,245500*3.154e7,75380*3.154e7,1600*3.154e7,
             3.8235*24*3600,3.10*60,26.8*60,19.9*60,164.3e-6,22.3*3.154e7,5.015*3.154e7,138.376*24*3600]
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]/half_life[0]
    dydx[1]=dydx[0]-y[1]/half_life[1]
    dydx[2]=dydx[1]-y[2]/half_life[2]
    dydx[3]=dydx[2]-y[3]/half_life[3]
    dydx[4]=dydx[3]-y[4]/half_life[4]
    dydx[5]=dydx[4]-y[5]/half_life[5]
    dydx[6]=dydx[5]-y[6]/half_life[6]
    dydx[7]=dydx[6]-y[7]/half_life[7]
    dydx[8]=dydx[7]-y[8]/half_life[8]
    dydx[9]=dydx[8]-y[9]/half_life[9]
    dydx[10]=dydx[9]-y[10]/half_life[10]
    dydx[11]=dydx[10]-y[11]/half_life[11]
    dydx[12]=dydx[11]-y[12]/half_life[12]
    dydx[13]=dydx[12]-y[13]/half_life[13]
    dydx[14]=dydx[13]
    return dydx


y0=np.zeros(15)
y0[0]=1
x0=0
x1=100

t1=time.time()
ans_stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau')
print(ans_stiff)
print(ans_stiff.t)
t2=time.time()
print('took ',ans_stiff.nfev,' evaluations and ',t2-t1,' seconds to solve implicitly')
print('final value is ',ans_stiff.y[0,-1],' with truth ')
