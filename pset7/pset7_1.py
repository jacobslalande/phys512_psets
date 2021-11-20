import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

'''
in the file PRNG_plane.png, we can clearly see plane by changing the perspective
of a 3d plot of the x,y,z coordinates

Python's PRNG doesnt show planes when plotted in 3d and the perspective is changed

Note that I wasnt able to do same on my local machine
'''

#C PRNG
rand_points = np.loadtxt('rand_points.txt')

x = rand_points[:,0]
y = rand_points[:,1]
z = rand_points[:,2]

a = 0
b = 2000


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(x[a:b], y[a:b], z[a:b], '.')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#python PRNG
u = np.random.rand(len(x),3)

xx = u[:,0]
yy = u[:,1]
zz = u[:,2]

a = 0
b = 3000

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(xx[a:b], yy[a:b], zz[a:b], '.')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
