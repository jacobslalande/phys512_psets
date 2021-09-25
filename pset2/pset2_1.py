import numpy as np
from matplotlib import pyplot as plt


#true answer : z>R -->1/z**2
#              z<R -->0

R = 20
z =  50
x = np.linspace(-1,1,2000)
#y = (z-R*x)/((R**2 + z**2 -2*R*z*x)**(3/2))
sign = np.sign(R**2 + z**2 -2*R*z*x)

y =(R**2)*(z-R*x)/((R**2 + z**2 -2*R*z*x)**(3/2))


R = 0.5
#z = np.linspace(R+1e-7,10,2000)
z = 0.5

y = (R**2)*(z-R*x)/((R**2 + z**2 -2*R*z*x)**(3/2))

E_true = 1/z**2

plt.plot(x,y)
plt.show()


