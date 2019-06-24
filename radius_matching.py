# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:47:12 2019

@author: jedhm
"""


import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

# https://github.com/kbarbary/sfdmap
import sfdmap
import extinction

from matplotlib import pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.coordinates import match_coordinates_sky
from astropy.cosmology import FlatLambdaCDM

from astroquery.irsa_dust import IrsaDust

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

os.path.abspath(os.curdir)
os.chdir("..")

c = 3e8

path_DR7 = 'data/DR7Q.fit'
t_DR7 = Table.read(path_DR7)
path_DR7Q = 'data/DR7QSOdata.fits'
t_DR7Q = Table.read(path_DR7Q)  
path_DR12 = 'data/DR12Q.fits'
t_DR12 = Table.read(path_DR12)
path_3XMM = 'data/3XMM_DR7cat.fits'
t_3XMM = Table.read(path_3XMM)

# BAD QUASAR REMOVAL, SEE FLAG DESCRIPTIONS
# REMOVE BAL_FLAG != 0 quasars () in SDSS DR7 AND radio loud quasars in SDSS DR7
t_DR7 = t_DR7Q[(t_DR7Q['BAL_FLAG'] == 0) & (t_DR7Q['R_6CM_2500A'] < 10)]
# DR12 BAL removal (removes ~ 10%) and GOOD detections flag
t_DR12 = t_DR12[(t_DR12['BAL_FLAG_VI'] == 0) & (t_DR12['CC_FLAGS'] == '0000')]
# DR12 Photometric quality
#combos = [''.join(i) for i in itertools.product('AB', repeat = 4)] # all acceptable flags combos
#t_DR12 = t_DR12[(t_DR12['PH_FLAG'] == 'BBBB')]

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
    # indices into catalog of matches, their 2d sep, 3d sep
    idx, d2d, d3d = coords_init.match_to_catalog_sky(catalog)
    
    # closest objects in radius of 2"
    mask = d2d < radius*u.arcsec
    
    # selection of matches in supplied coordinates
    coords, coords_errs = coords_init[mask], coords_errs_init[idx[mask]]
    coords_cat, coords_cat_errs = catalog[idx[mask]], coords_errs_init[idx[mask]]
    
    # return coordinates of matches in coords_init, catalog and their errors
    # as well as indices into catalog of matches < 2" away.
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
        
        # coordinate separation from every other coordinate within error radius.
        catalogmsk = c.separation(coords) < c_err*u.arcsec
        idxcatalog = np.where(catalogmsk)[0]
        
        # indices that have separation within error radius in catalog
        catalog_idxmsk = cat_mask[idxcatalog]
        
        coords_matches.append(coords[idxcatalog])
        coords_matches_err.append(coords_errs[idxcatalog])
        cat_idx_matches.append(catalog_idxmsk)
        
        
        print('coord: ', coords[idxcatalog])
        print('count: ', len(coords_matches)) 
    
    return SkyCoord(np.concatenate(coords_matches)), np.concatenate(coords_matches_err), np.concatenate(cat_idx_matches)
    

# inital matches with 2" radius

DR7_coords_init, DR7_coords_errs_init, XMMa_coords_init, XMMa_coords_errs_init, cat_maskDR7 = initial_match(coords_DR7, coords_3XMM, dpos1, 2.0)
DR12_coords_init, DR12_coords_errs_init, XMMb_coords_init, XMMb_coords_errs_init, cat_maskDR12 = initial_match(coords_DR12, coords_3XMM, dpos1, 2.0)

# matched coordinates SDSS->3XMM (dyn. errs.)
DR7_matches, DR7_matches_poserr, XMM_DR7_idx_matches = radius_match(DR7_coords_init, DR7_coords_errs_init, cat_maskDR7)
DR12_matches, DR12_matches_poserr, XMM_DR12_idx_matches = radius_match(DR12_coords_init, DR12_coords_errs_init, cat_maskDR12)

# matched back into DR7,DR12 coords
#idx_back, d2d_back, d3d_back = DR7_coords_init.match_to_catalog_sky(coords_DR7)
idx_bk_DR7, d2d_bk_DR7, d3d_bk_DR7 = DR7_matches.match_to_catalog_sky(coords_DR7)
idx_bk_DR12, d2d_bk_DR12, d3d_bk_DR12 = DR12_matches.match_to_catalog_sky(coords_DR12)

mask1, mask2 = d2d_bk_DR7 < 0.01*u.arcsec, d2d_bk_DR12 < 0.01*u.arcsec

# %% matches to tables + back matching

# any point in having coordinates????? just use indexes out of functions ?

# 3XMM-DR7 table matching
t_3XMM_DR7_matches = t_3XMM[XMM_DR7_idx_matches[(DR7_matches_poserr < 1.5)]]
t_3XMM_DR12_matches = t_3XMM[XMM_DR12_idx_matches[(DR12_matches_poserr < 1.5)]]

# DR7, DR12 back matching IMPLEMENT THIS IN FUNCTIONS
t_DR7_DR7_matches = t_DR7[idx_bk_DR7[(DR7_matches_poserr < 1.5)]]
t_DR12_DR12_matches = t_DR12[idx_bk_DR12[(DR12_matches_poserr < 1.5)]]

print('Quasars 3XMM->SDSS: ', len(t_3XMM_DR7_matches) + len(t_3XMM_DR12_matches))
print('Quasars2 SDSS->SDSS: ', len(t_DR7_DR7_matches) + len(t_DR12_DR12_matches))