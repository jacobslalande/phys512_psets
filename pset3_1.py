import numpy as np
from matplotlib import pyplot as plt

j = 0

def fun(x,y):
    dydx = y/(1+x**2)
    return dydx

def rk4_step(fun,x,y,h):
    k1=fun(x,y)*h
    k2=h*fun(x+h/2,y+k1/2)
    k3=h*fun(x+h/2,y+k2/2)
    k4=h*fun(x+h,y+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    return dy

def rk4_stepd(fun,x,y,h):
    '''
    the return value is found by using y1_(n+1) = y_true + h**5 for step h
    and y2_(n+1) = y_true + 2*(h**5)/32 for step h/2
    The 2 in front of O(h**2) for y2 comes from the fact that y2 contains two steps of h/2
    and the error adds.
    Then by comparing y1 and y2 find h**5, plug in y1, isolate y_true and have relation
    returned below
    '''
    #compute with step h
    rk4_h = rk4_step(fun,x,y,h)
    
    #compute with step h/2 to find y in middle of range (x,x+h)
    rk4_1 = rk4_step(fun,x,y,h/2)
    
    #compute y middle
    y_mid = y + rk4_1
    
    #compute other step h/2 that leads to same y as for step h
    rk4_2 = rk4_step(fun,x+ h/2,y_mid,h/2)
    
    #compute same y as for step h using step h/2
    y_final = y_mid + rk4_2
    
    # compute y_n+1 with step h
    y1 = y + rk4_h
        
    return y1 + (16/15)*(y_final-y1)

nstep=200 # number of steps
x=np.linspace(-20,20,nstep+1) #range of x

y_step, y_stepd = 0*x, 0*x # create array with same length as x for each rk4 integrator
y_step[0], y_stepd[0] = 1, 1 # initial value is y(-20)=1

y_pred=np.exp(np.arctan(x)+np.arctan(20)) # true function for y(x)

k = 0
for i in range(nstep):
    h=x[i+1]-x[i] # define step
    y_step[1+i] = y_step[i] + rk4_step(fun,x[i],y_step[i],h) # first integrator
    y_stepd[1+i] = rk4_stepd(fun,x[i],y_stepd[i],h) #second integrator
    k = k+12
    
print('num of steps second stepper :',k,'\nnum of steps first stepper :',4*i,'\nnumber of steps such that same number of fcn eval for 2nd stepper :',4*i/12)
    
if True :
    nstep=int(4*i/12) # number of steps
    x1=np.linspace(-20,20,nstep+1) #range of x
    y_pred1=np.exp(np.arctan(x1)+np.arctan(20)) # true function for y(x)

    y_stepd = 0*x1 # create array with same length as x for each rk4 integrator
    y_stepd[0] =  1 # initial value is y(-20)=1

    for i in range(nstep):
        h=x1[i+1]-x1[i] # define step
        y_stepd[1+i] = rk4_stepd(fun,x1[i],y_stepd[i],h) #second integrator

plt.plot(x,y_step,'r')
plt.plot(x1,y_stepd)
plt.plot(x,y_pred,'*')
print('error first integrator : ',np.std(y_step-y_pred),'\nerror second integrator : ', np.std(y_stepd-y_pred1))
plt.show()