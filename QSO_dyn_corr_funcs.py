# %% initial

import os
import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.coordinates import match_coordinates_sky

os.path.abspath(os.curdir)
os.chdir("..")

path_DR7 = 'data/DR7Q.fit'
t_DR7 = Table.read(path_DR7)
path_DR7Q = 'data/DR7QSOdata.fits'
t_DR7Q = Table.read(path_DR7Q)  
path_DR12 = 'data/DR12Q.fits'
t_DR12 = Table.read(path_DR12)
path_3XMM = 'data/3XMM_DR7cat.fits'
t_3XMM = Table.read(path_3XMM)

# REMOVE BAL_FLAG != 0 quasars () in SDSS DR7 AND radio loud quasars in SDSS DR7
t_DR7 = t_DR7Q[(t_DR7Q['BAL_FLAG'] == 0) & (t_DR7Q['R_6CM_2500A'] < 10)]
# DR12 BAL removal (removes ~ 10%)
t_DR12 = t_DR12[(t_DR12['BAL_FLAG_VI'] == 0)]

# %% Dynamic match radius correlation

# dynamic sep. constraints: 1arcsec sym. err. for SDSS cats. and object pos errs for 3XMM
dpos1 = np.sqrt(np.square(np.array(t_3XMM['SC_POSERR'])) + np.square(np.full((499266), 0.0*u.arcsec)))

coords_DR7 = SkyCoord(t_DR7['RA']*u.deg, t_DR7['DEC']*u.deg)
coords_DR12 = SkyCoord(t_DR12['RA'], t_DR12['DEC'])
coords_3XMM = SkyCoord(t_3XMM['SC_RA'], t_3XMM['SC_DEC'])


def initial_match(coords_init, catalog, coords_errs_init, radius):
    """ 
        Initial match of radius k arcsec to reduce data set.
        requires coords and a catalog as arrays of RA,DEC coordinates
        and their quadrature positional errors as an array. 
        
        Returns: coords - matched objects in supply coordinates 
                 coords_cat - matched objects in catalog
                 idx[] - indices in catalog of matches
    """
    idx, d2d, d3d = coords_init.match_to_catalog_sky(catalog)
    
    mask = d2d < radius*u.arcsec
    
    coords, coords_errs = coords_init[mask], coords_errs_init[idx[mask]]
    coords_cat, coords_cat_errs = catalog[idx[mask]], coords_errs_init[idx[mask]]

    return coords, coords_errs, coords_cat, coords_cat_errs, idx[mask]

def radius_match(coords,coords_errs,cat_mask):
    """
        Derives set of coordinates within a given set of coordinates 
        with separation from catalog matches less than their 
        quadrature positional errors.
    """
    coords_matches = [] 
    coords_matches_err = []
    cat_idx_matches = []

    for c, c_err in np.column_stack((coords,coords_errs)):

        catalogmsk = c.separation(coords) < c_err*u.arcsec
        idxcatalog = np.where(catalogmsk)[0]
    
        catalog_idxmsk = cat_mask[idxcatalog]
    
        coords_matches.append(coords[idxcatalog])
        coords_matches_err.append(coords_errs[idxcatalog])
        cat_idx_matches.append(catalog_idxmsk)
        
        print('mask ',sum(catalogmsk))
        print('idx_cat',sum(idxcatalog))
        #print(coords[idxcatalog])
        print(len(coords_matches)) 
    
    return np.concatenate(coords_matches), np.concatenate(coords_matches_err), cat_idx_matches

def error_filter_count(source_errs, bound):
    
    """
        number of sources with error < bound
    """
    count = 0
    other = 0
    for obj in source_errs:
        if obj < 1.5:
            count += 1
        else:
            other += 1
            
    return count

# inital matches with 2" radius

DR7_coords_init, DR7_coords_errs_init, XMMa_coords_init, XMMa_coords_errs_init, cat_maskDR7 = initial_match(coords_DR7, coords_3XMM, dpos1, 2.0)
DR12_coords_init, DR12_coords_errs_init, XMMb_coords_init, XMMb_coords_errs_init, cat_maskDR12 = initial_match(coords_DR12, coords_3XMM, dpos1, 2.0)

# matched coordinates SDSS->3XMM (dyn. errs.)
DR7_matches, DR7_matches_poserr, XMM_DR7_idx_matches = radius_match(DR7_coords_init, DR7_coords_errs_init, cat_maskDR7)
DR12_matches, DR12_matches_poserr, XMM_DR12_idx_matches = radius_match(DR12_coords_init, DR12_coords_errs_init, cat_maskDR12)
# matched coordinates SDSS->3XMM->3XMM (dyn. errs.) (ARE THESE NEEDED NOW ONE HAS XMM_DR#_idx_matches)
XMMa_matches, XMMa_matches_poserr = radius_match(XMMa_coords_init, XMMa_coords_errs_init)
XMMb_matches, XMMb_matches_poserr = radius_match(XMMb_coords_init, XMMb_coords_errs_init)

count_DRcats = error_filter_count(DR7_matches_poserr,1.5) + error_filter_count(DR12_matches_poserr,1.5)


# %% spherical plots
#
## Create a sphere
#r, pi = 1, np.pi
#phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
#x = r*np.sin(phi)*np.cos(theta)
#y = r*np.sin(phi)*np.sin(theta)
#z = r*np.cos(phi)
#
## func for spherical coords of catalogs
#def coords_to_sphericals(coords):
#    theta, phi = coords.ra.rad, coords.dec.rad
#    xx = np.sin(phi)*np.cos(theta)
#    yy = np.sin(phi)*np.sin(theta)
#    zz = np.cos(phi)
#
#    return xx,yy,zz
#
#def coords_to_sphericals1(coords):
#    theta, phi = coords, coords
#    xx = np.sin(phi)*np.cos(theta)
#    yy = np.sin(phi)*np.sin(theta)
#    zz = np.cos(phi)
#
#    return xx,yy,zz
#
#
#x1,y1,z1 = coords_to_sphericals(coords_3XMM)
#x1a,y1a,z1a = coords_to_sphericals1(XMMa_matches)
#x1c,y1c,z1c = coords_to_sphericals1(XMMb_matches)
#x2,y2,z2 = coords_to_sphericals(coords_DR7)
#x2a,y2a,z2a = coords_to_sphericals1(DR7_matches)
#x3,y3,z3 = coords_to_sphericals(coords_DR12)
#x3a,y3a,z3a = coords_to_sphericals1(DR12_matches)
#
#fig, axs = plt.subplots(3)
#axs[0].plot_surface(x, y, z,  rstride=1, cstride=1, color='dimgray', alpha=0.15, linewidth=0)
#axs[0].scatter(x1,y1,z1,color="plum",s=0.001)
#axs[0].scatter(x1a,y1a,z1a,color="lightsalmon",s=0.001)
#axs[0].scatter(x1a,y1a,z1a,color="lightsalmon",s=0.001)
#axs[0].set_title('')
#axs[1].plot_surface(x, y, z,  rstride=1, cstride=1, color='dimgray', alpha=0.15, linewidth=0)
#axs[1].scatter(x2,y2,z2,color="plum",s=0.05)
#axs[1].scatter(x2a,y2a,z2a,color="lightsalmon",s=0.05)
#axs[1].set_title('')
#axs[2].plot_surface(x, y, z,  rstride=1, cstride=1, color='dimgray', alpha=0.15, linewidth=0)
#axs[2].scatter(x3,y3,z3,color="plum",s=0.05)
#axs[2].scatter(x3a,y3a,z3a,color="lightsteelblue",s=0.05)
#axs[2].set_title('')
#
#
#axs.axis('off')
#axs.set_xlim([-1,1])    
#axs.set_ylim([-1,1])
#axs.set_zlim([-1,1])
#axs.set_facecolor('darkgrey') 

#%%
