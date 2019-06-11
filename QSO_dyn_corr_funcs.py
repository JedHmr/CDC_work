# %% initialíí

import os
import numpy as np

# https://github.com/kbarbary/sfdmap
import sfdmap

from matplotlib import pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.coordinates import match_coordinates_sky

from astroquery.irsa_dust import IrsaDust

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
        
        print('coord: ',coords[idxcatalog])
        print('count: ',len(coords_matches)) 
    
    return np.concatenate(coords_matches), np.concatenate(coords_matches_err), np.concatenate(cat_idx_matches)


# inital matches with 2" radius

DR7_coords_init, DR7_coords_errs_init, XMMa_coords_init, XMMa_coords_errs_init, cat_maskDR7 = initial_match(coords_DR7, coords_3XMM, dpos1, 2.0)
DR12_coords_init, DR12_coords_errs_init, XMMb_coords_init, XMMb_coords_errs_init, cat_maskDR12 = initial_match(coords_DR12, coords_3XMM, dpos1, 2.0)

# matched coordinates SDSS->3XMM (dyn. errs.)
DR7_matches, DR7_matches_poserr, XMM_DR7_idx_matches = radius_match(DR7_coords_init, DR7_coords_errs_init, cat_maskDR7)
DR12_matches, DR12_matches_poserr, XMM_DR12_idx_matches = radius_match(DR12_coords_init, DR12_coords_errs_init, cat_maskDR12)

# %% matches to tables back matching

# any point in having coordinates????? just use indexes out of functions ?

# 3XMM-DR7 table matching
t_3XMM_DR7_matches = t_3XMM[XMM_DR7_idx_matches[(DR7_matches_poserr < 1.5)]]
t_3XMM_DR12_matches = t_3XMM[XMM_DR12_idx_matches[(DR12_matches_poserr < 1.5)]]

# DR# table matching
t_DR7_DR7_matches = t_DR7[t_DR7['RA'] == coords_DR7.ra.deg]
t_DR12_DR12_matches = t_DR12[t_DR12['RA'] == coords_DR12.ra.deg]

print('Quasars: ', len(t_3XMM_DR7_matches) + len(t_3XMM_DR12_matches))

# %% XRAY FLUX 
#DR7,DR12 objs with 3XMM xray measures
FX_soft_DR7 = t_3XMM_DR7_matches['SC_EP_2_FLUX'] + t_3XMM_DR7_matches['SC_EP_3_FLUX']
FX_soft_err_DR7 = t_3XMM_DR7_matches['SC_EP_2_FLUX_ERR'] + t_3XMM_DR7_matches['SC_EP_3_FLUX_ERR']

FX_hard_DR7 = t_3XMM_DR7_matches['SC_EP_4_FLUX'] + t_3XMM_DR7_matches['SC_EP_5_FLUX']
FX_hard_err_DR7 = t_3XMM_DR7_matches['SC_EP_4_FLUX_ERR'] + t_3XMM_DR7_matches['SC_EP_5_FLUX_ERR']

FX_soft_DR12 = t_3XMM_DR12_matches['SC_EP_2_FLUX'] + t_3XMM_DR12_matches['SC_EP_3_FLUX']
FX_soft_err_DR12 = t_3XMM_DR12_matches['SC_EP_2_FLUX_ERR'] + t_3XMM_DR12_matches['SC_EP_3_FLUX_ERR']

FX_hard_DR12 = t_3XMM_DR12_matches['SC_EP_4_FLUX'] + t_3XMM_DR12_matches['SC_EP_5_FLUX']
FX_hard_err_DR12 = t_3XMM_DR12_matches['SC_EP_4_FLUX_ERR'] + t_3XMM_DR12_matches['SC_EP_5_FLUX_ERR']


#%% UV FLUX

# SFD (1998) extinction map --> galactic extinction
m = sfdmap.SFDMap('C:/Users/jedhm/Documents/Projects/CDC_work/data/sfd_data')

R_v = 3.1

# DR7 UV
UV_DR7 = t_DR7_DR7_matches['LOGFNU2500A_ERGS_OBS']
z_DR7 = t_DR7_DR7_matches['REDSHIFT']

# objectwise galactic extinction (average with matched 3XMM coords?)
ext_DR7 = R_v*m.ebv(coords_DR7) # R_v*E(B-V) = A_v
ext_DR12 = R_v*m.ebv(coords_DR12) # R_v*E(B-V) = A_v

# DR12 UV
# redshifts 'z' 
z_DR12 = t_DR12_DR12_matches['Z_CIV']

# Y,J,K,H DR12 flux, filter by SNR?
fY, fJ = t_DR12_DR12_matches['YFLUX'], t_DR12_DR12_matches['JFLUX']
fH, fK = t_DR12_DR12_matches['HFLUX'], t_DR12_DR12_matches['KFLUX']

fY_err, fJ_err = t_DR12_DR12_matches['ERR_YFLUX'], t_DR12_DR12_matches['ERR_JFLUX'] 
fH_err, fK_err = t_DR12_DR12_matches['ERR_HFLUX'], t_DR12_DR12_matches['ERR_KFLUX'] 

# eff. freqs 
Jfreq, Hfreq, Kfreq = 4.16e-18, 5.44e-15, 7.34e-15

# coord targets
coords_DR7 = SkyCoord(t_DR7_DR7_matches['RA']*u.deg, t_DR7_DR7_matches['DEC']*u.deg)
coords_DR12 = SkyCoord(t_DR12_DR12_matches['RA'], t_DR12_DR12_matches['DEC'])

# add extinction to object mags CONVERT TO FLUXES ?
J, H, K = J + ext_DR12, H + ext_DR12, K + ext_DR12




def mag_to_flux(mags_SDSS, mag_type):
    """
        asinh() mags to flux density for given SDSS band magnitude
        i.e. SDSS ugriz mags --> AB mags --> flux density
        
        f_0: conventional magnitude = 0 amount of flux
        b: softening factor
        
        ***Returns flux in Jy***
    """

    if mag_type == 'ugriz':
        #softening factors 'band' : (softening factor,zero-flux mag.)
        filters = {
                'u' : (1,1.4e-10,24.63,0.04),
                'g' : (2,0.9e-10,25.11,0),
                'r' : (3,1.2e-10,24.80,0),
                'i' : (4,1.8e-10,24.36,0),
                'z' : (5,7.4e-10,22.83,0.02)
                }
    
        # function for asinh mag to f/f_0 (fraction of AB zeropoint flux density)
        def F_F0(mags,key):
            return np.sinh(((np.log(10)*mags)/(-2.5) - np.log(filters[key][1])))
        
        # space for flux
        S_AB = np.zeros_like(t_DR12_DR12_matches['PSFMAG'])
        
        # each mags col has its own SDSS->AB conversion
        for key in filters.keys():
            for i in range(0,np.shape(mags_SDSS)[1]):
                mags_SDSS[:,i] = mags_SDSS[:,i] - filters[key][3]
                S_AB[:,i] = 3631*F_F0(mags_SDSS[:,i], key)
    
        return S_AB
    
    else:
        print('this is only for SDSS ugriz luptitudes')

#%%

