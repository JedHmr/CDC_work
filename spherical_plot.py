# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:38:11 2019

@author: jedhm
"""

%% spherical plots

csDR7 = SkyCoord(t_3XMM_DR7_matches['SC_RA'],t_3XMM_DR7_matches['SC_DEC'])
csDR12 =  SkyCoord(t_3XMM_DR12_matches['SC_RA'],t_3XMM_DR12_matches['SC_DEC'])

# Create a sphere
r, pi = 1, np.pi
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x = r*np.sin(phi)*np.cos(theta)
y = r*np.sin(phi)*np.sin(theta)
z = r*np.cos(phi)

# func for spherical coords of catalogs
def coords_to_sphericals(coords):
    theta, phi = coords.ra.rad, coords.dec.rad
    xx = np.sin(phi)*np.cos(theta)
    yy = np.sin(phi)*np.sin(theta)
    zz = np.cos(phi)
    
    return xx, yy, zz
    

# catalogs
x1,y1,z1 = coords_to_sphericals(coords_3XMM)
# matches 
x1a,y1a,z1a = coords_to_sphericals(csDR7)
x1b,y1b,z1b = coords_to_sphericals(csDR7)

fig, ax = plt.subplots(1)
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0.1)
ax.scatter(x1,y1,z1,color="slategrey",s=0.001,zorder=0)
ax.scatter(x1a,y1a,z1a,color="r",s=0.2,zorder=500)
ax.scatter(x1b,y1b,z1b,color="b",s=0.2,zorder=200)

# ax.scatter(coords_to_sphericals(coords_3XMM),color="slategrey",s=0.001)
# ax.scatter(coords_to_sphericals(csDR7),color="r",s=0.001)
# ax.scatter(coords_to_sphericals(csDR12),color="b",s=0.001)

ax.axis('off')
ax.set_aspect('equal')
ax.set_xlim([-1,1])    
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_facecolor('darkgrey')
