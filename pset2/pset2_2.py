import numpy as np
import time
def lorentz(x):
    return 1/(1+x**2)


counter = 0
counter_ = 0
def integrate_adaptive(fun,a,b,tol,extra=None):

    #hardwire to use simpsons
    #note it would be more useful if i generalized the number of points in x to be input
    x=np.linspace(a,b,5)
    
    global counter #counts number of function call for this fcn
    global counter_ # counts number of function call for lazy way
    
    
    dx=(b-a)/(len(x)-1)
    
    if extra == None:
        y=fun(x)
        y_0,y_2,y_4 = y[0],y[2],y[4]
        area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
        area2=dx*(y_0+4*y[1]+2*y_2+4*y[3]+y_4)/3 #finer step
        err=np.abs(area1-area2)
        
        counter,counter_ = 5,5
        
    elif extra[-1] == 0:
        #left
        
        #new values
        y_new1 = fun(x[1])        
        y_new2 = fun(x[3])
        
        counter = counter + 2
        counter_ += 5
        
        #stored values      
        y_0 = extra[0] 
        y_mid = extra[1]
        y_last = extra[2]
        
        #compute area
        area1 = 2*dx*(y_0+4*y_mid+y_last)/3
        area2 = dx*(y_0+4*y_new1+2*y_mid+4*y_new2+y_last)/3
        err=np.abs(area1-area2)
        
    else:
        #right
        
        #new values
        y_new1 = fun(x[1])
        y_new2 = fun(x[3])        
        
        counter += 2
        counter_ += 5
        
        #stored values
        y_0 = extra[0] 
        y_mid = extra[1]
        y_last = extra[2]
        
        area1 = 2*dx*(y_0+4*y_mid+y_last)/3
        area2 = dx*(y_0+4*y_new1+2*y_mid+4*y_new2+y_last)/3
        
        err=np.abs(area1-area2)
    
    if err<tol:
        return area2
    else:
        #if miss tolerance, want smaller tol
        xmid=(a+b)/2 
        
        if extra == None:
            extra_l = [y[0],y[1],y[2],0]
            extra_r = [y[2],y[3],y[4],1]
        else:
            #store value computed which will be necessary in next step
            extra_l = [y_0,y_new1,y_mid,0]
            extra_r = [y_mid,y_new2,y_last,1]
            
        left=integrate_adaptive(fun,a,xmid,tol/2,extra_l)
        right=integrate_adaptive(fun,xmid,b,tol/2,extra_r)
        return left+right


a=-10
b=10

if False:
    ans=integrate_adaptive(np.exp,a,b,1e-7)
    print(ans-(np.exp(b)-np.exp(a)))
else:
    ans=integrate_adaptive(lorentz,a,b,1e-3)
    print(ans-(np.arctan(b)-np.arctan(a)))

print('counting of function call for new fcn and lazy way, resp. :',counter,counter_)