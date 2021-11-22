import numpy as np
from matplotlib import pyplot as plt


N=10000000
def ratio_exp(b) : 
    # Compute bounds
    '''
    found by isolating v from the inequality 0<=u<=sqrt(p(u/v)), 
    then maximizing v as a fcn of u to find u maximizing v
    
    bounds for v : 0 <= v <= 2/(np.exp(1)*np.sqrt(b))
    '''
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

print(f"Method is {len(t)/(2*N) * 100}% Efficient")
bins=np.linspace(0,20,101) #bin the events
aa,bb=np.histogram(t,bins) #bb gives bin edges, aa gives the real pdf
cents=(bb[1:]+bb[:-1])/2 #want center of bins
pred=b*np.exp(-b*cents) #pdf should go like this
aa=aa/aa.sum() #normalize
pred=pred/pred.sum()

# Plotting Results
plt.figure()
plt.hist(t,bins=50,density=True, label="Samples")
xs = np.linspace(0,max(t),1000)
plt.plot(xs,b*np.exp(-b*xs),label="Expected")
plt.legend()
plt.xlabel(f"Method is {len(t)/(2*N) * 100}% Efficient")
plt.savefig('q3_hist.png')
plt.show()
