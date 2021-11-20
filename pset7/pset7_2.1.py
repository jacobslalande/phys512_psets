import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf


N=10000000
def ratio_exp(b) : 
    # Calculated Bounds 
    umax = np.sqrt(b)
    vmax = 2/(np.exp(1)*np.sqrt(b))


    # Generate the random numbers
    u=np.random.uniform(low=0,high=umax,size=N)
    v=np.random.uniform(low=0,high=vmax,size=N)

    r = v/u

    # Do the rejection Step
    keep = u < b*np.exp(-b*r/2)
    t = r[keep]
    return t
b = 1
t = ratio_exp(b)
'''
# Plot Results
plt.plot()
# Cut far numbers to get a better plot.
plt.hist(t,bins=3000,density=True)
xs = np.linspace(0,5,1000)
plt.plot(xs,b*np.exp(-b*xs))
plt.xlim(min(xs),max(xs))
print(f"Method is {len(t)/(2*len(u)) * 100}% Efficient")
plt.show()
'''
print(f"Method is {len(t)/(2*N) * 100}% Efficient")
bins=np.linspace(0,20,101) #bin the events
aa,bb=np.histogram(t,bins) #bb gives bin edges, aa gives the real pdf
cents=(bb[1:]+bb[:-1])/2 #want center of bins
pred=b*np.exp(-b*cents) #pdf should go like this
aa=aa/aa.sum() #normalize
pred=pred/pred.sum()

plt.plot(cents,aa,'*')
plt.plot(cents,pred,'r')
plt.show()