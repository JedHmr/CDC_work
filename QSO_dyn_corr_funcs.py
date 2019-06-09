# %% initial

import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.coordinates import match_coordinates_sky

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
    """
    idx, d2d, d3d = coords_init.match_to_catalog_sky(catalog)

    coords, coords_errs = coords_init[d2d < radius*u.arcsec], coords_errs_init[idx[d2d < radius*u.arcsec]]
    coords_cat, coords_cat_errs = catalog[idx[d2d < radius*u.arcsec]], coords_errs_init[idx[d2d < radius*u.arcsec]]

    return coords, coords_errs, coords_cat, coords_cat_errs

def radius_match(coords,coords_errs):
    """
        Derives set of coordinates within set of coordinates with 
        separation from catalog matches less than their quadrature
        positional errors.
    """
    coords_matches = [] 
    coords_matches_err = []

    for c, c_err in np.column_stack((coords,coords_errs)):

        catalogmsk1 = c.separation(coords) < c_err*u.arcsec
        idxcatalog1 = np.where(catalogmsk1)[0]  
    
        coords_matches.append(coords[idxcatalog1])
        coords_matches_err.append(coords_errs[idxcatalog1])

        print(coords[idxcatalog1])
        print(len(coords_matches)) 
    
    return np.concatenate(coords_matches), np.concatenate(coords_matches_err)

# inital matches with 2" radius
DR7_coords_init, DR7_coords_errs_init, XMMa_coords_init, XMMa_coords_errs_init = initial_match(coords_DR7, coords_3XMM, dpos1, 2.0)
DR12_coords_init, DR12_coords_errs_init, XMMb_coords_init, XMMb_coords_errs_init = initial_match(coords_DR12, coords_3XMM, dpos1, 2.0)

# matched coordinates SDSS->3XMM (dyn. errs.)
DR7_matches, DR7_matches_poserr = radius_match(DR7_coords_init, DR7_coords_errs_init)
DR12_matches, DR12_matches_poserr = radius_match(DR12_coords_init, DR12_coords_errs_init)
# matched coordinates SDSS->3XMM->3XMM (dyn. errs.)
XMMa_matches, XMMa_matches_poserr = radius_match(XMMa_coords_init, XMMa_coords_errs_init)
XMMb_matches, XMMb_matches_poserr = radius_match(XMMb_coords_init, XMMb_coords_errs_init)
