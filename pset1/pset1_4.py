import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

npt = 5
#note that if npt != 5 then the rational fit is garbage

#define spline
def spline(x,y,xx):
    spln=interpolate.splrep(x,y)
    yy=interpolate.splev(xx,spln)
    return yy

#deifne polynomial
def poly(x,y,xx):
    X=np.empty([npt,npt])

    for i in range(npt):
        X[:,i]=x**i
    Xinv=np.linalg.inv(X)
    c=Xinv@y
    y_pred=X@c

    
    XX=np.empty([len(xx),npt])
    for i in range(npt):
        XX[:,i]=xx**i
    yy=XX@c
    return yy

#define rational interpolation
def rateval(x,p,q):
    top=0
    for i,par in enumerate(p):
        top=top+par*x**i # for every order take x**i and multiply with
                        # corresponding parameter
    bot=1
    for i,par in enumerate(q):
        bot=bot+par*x**(i+1)
    return top/bot

def ratfit(y,x,n,m):
    npt=len(x)
    assert(len(y)==npt)
    assert(n>=0)
    assert(m>=0)
    assert(n+1+m==npt) #need total number of coef=# of data pts (for inverting)

    top_mat=np.empty([npt,n+1]) #X, n+1 to include order 0 (not square)
    bot_mat=np.empty([npt,m]) #X'
    
    #fill the matrices
    for i in range(n+1):
        top_mat[:,i]=x**i
    for i in range(m):
        bot_mat[:,i]=y*x**(i+1)
        
    mat=np.hstack([top_mat,-bot_mat])

    pars=np.linalg.inv(mat)@y
    p=pars[:n+1]
    q=pars[n+1:]
    return p,q

def rat(x,y,xx,order=False):
    if order == True:
        n,m = 4,5
    else:
        m=len(y)//2
        n=len(y)-m-1
    p,q = ratfit(y,x,n,m)
    yy = rateval(xx,p,q)
    
    return yy

def accuracy(x,y,xx,fun):
    yy = fun(xx)
    print('spline : ',np.std(spline(x,y,xx)-yy))
    print('polynomial : ',np.std(poly(x,y,xx)-yy))
    print('rational : ',np.std(rat(x,y,xx)-yy))
    print('\n')


#compare accuracy for cos
xmin = -np.pi/2
xmax = np.pi/2

x = np.linspace(xmin,xmax,npt)
y = np.cos(x)
xx=np.linspace(xmin,xmax,2001)
yy = np.cos(xx)

accuracy(x,y,xx,np.cos)

#compare accuracy for lorentzian
xmin = -1
xmax = 1

x = np.linspace(xmin,xmax,npt)
y = 1/(1+x**2)
xx=np.linspace(xmin,xmax,2001)
yy = 1/(1+xx**2)

fun = lambda x : 1/(1+x**2)

accuracy(x,y,xx,fun)

