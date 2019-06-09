import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.coordinates import match_coordinates_sky


# data
path_DR7 = 'data/DR7Q.fit'
t_DR7 = Table.read(path_DR7)
path_DR7Q = 'data/DR7QSOdata.fits'
t_DR7Q = Table.read(path_DR7Q)  # same table as DR7 for some reason?
path_DR12 = 'data/DR12Q.fits'
t_DR12 = Table.read(path_DR12)
path_3XMM = 'data/3XMM_DR7cat.fits'
t_3XMM = Table.read(path_3XMM)

coords_DR7 = SkyCoord(t_DR7['RA']*u.deg, t_DR7['DEC']*u.deg)
coords_DR12 = SkyCoord(t_DR12['RA'], t_DR12['DEC'])
coords_3XMM = SkyCoord(t_3XMM['SC_RA'], t_3XMM['SC_DEC'])

# Create a sphere
r = 1
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)


theta, phi = coords_3XMM.ra.rad, coords_3XMM.dec.rad
x1 = sin(phi)*cos(theta)
y1 = sin(phi)*sin(theta)
z1 = cos(phi)   
theta, phi = coords_DR7.ra.rad, coords_DR7.dec.rad
x2 = sin(phi)*cos(theta)
y2 = sin(phi)*sin(theta)
z2 = cos(phi)   
theta, phi = coords_DR12.ra.rad, coords_DR12.dec.rad
x3 = sin(phi)*cos(theta)
y3 = sin(phi)*sin(theta)
z3 = cos(phi)   

#Set colours and render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='dimgray', alpha=0.3, linewidth=0)

ax.scatter(x1,y1,z1,color="plum",s=0.001)
ax.scatter(x2,y2,z2,color="lightsteelblue",s=0.05)
ax.scatter(x3,y3,z3,color="powderblue",s=0.05)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.axis('off')
ax.set_facecolor('darkgrey')
#ax.set_aspect("equal")
plt.tight_layout()
plt.show()