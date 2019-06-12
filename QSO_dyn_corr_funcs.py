# %% initial

import os
import numpy as np

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

from astroquery.irsa_dust import IrsaDust

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

# %% matches to tables + back matching

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
R_v, m = 3.1, sfdmap.SFDMap('C:/Users/jedhm/Documents/Projects/CDC_work/data/sfd_data')

# redshifts 'z' 
z_DR7, z_DR12 = t_DR7_DR7_matches['REDSHIFT'], t_DR12_DR12_matches['Z_CIV']

# objectwise galactic extinction (average with matched 3XMM coords?)
ext_DR7 = R_v*m.ebv(coords_DR7) # R_v*E(B-V) = A_v
ext_DR12 = R_v*m.ebv(coords_DR12) # R_v*E(B-V) = A_v

# coord targets (refreshed)
coords_DR7 = SkyCoord(t_DR7_DR7_matches['RA']*u.deg, t_DR7_DR7_matches['DEC']*u.deg)
coords_DR12 = SkyCoord(t_DR12_DR12_matches['RA'], t_DR12_DR12_matches['DEC'])

# DR7 UV
UV_DR7 = t_DR7_DR7_matches['LOGFNU2500A_ERGS_OBS']

# DR12 UV

# add extinction to object mags CONVERT TO FLUXES ?
#J, H, K = J + ext_DR12, H + ext_DR12, K + ext_DR12

# ugriz DR12 mags minus columnwise extinction (v-band)
ugriz_mags1 = t_DR12_DR12_matches['PSFMAG'] - (np.array([ext_DR12]*5)).transpose()
# using five band extinction Schlegel et al. 1998 
ugriz_mags2 = t_DR12_DR12_matches['PSFMAG'] - t_DR12_DR12_matches['EXTINCTION_RECAL']
        
# WISE mags minus Vband extinction ?! INFRARED ?
WISE_mags = [t_DR12_DR12_matches['W{}MAG'.format(i)] - ext_DR12 for i in range(1,5)] 
WISE_mags = WISE_mags - (np.array([ext_DR12]))
WISE_mags = np.array(WISE_mags).transpose()

# def WISE_magvega_to_flux(mags):
#     # zero mag. flux densities for bands
#     # http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
#     f_0s = {
#             'w1': 
#             }
#     return


def mag_to_flux(mags, mag_type):
    """
        ugriz or YJHK type magnitudes
        
        asinh() mags to flux density for given SDSS band magnitude
        i.e. SDSS ugriz mags --> AB mags --> flux density
        
        f_0: conventional magnitude = 0 amount of flux
        b: softening factor
        
        ***Returns flux in Jy***
    """
    c=3e8
    # SDSS MAGS
    if mag_type == 'ugriz':
        # u,g,r,i,z freqs
        ugriz_freqs = np.array([c/354e-9, c/475e-9, c/622e-9, c/763e-9, c/905e-9])
        
        #softening factors 'band' : (softening factor,zero-flux mag.)
        filters = {
                'u' : (1,1.4e-10,24.63,0.04),
                'g' : (2,0.9e-10,25.11,0),
                'r' : (3,1.2e-10,24.80,0),
                'i' : (4,1.8e-10,24.36,0),
                'z' : (5,7.4e-10,22.83,0.02)
                }
    
        # function for asinh mag to f/f_0 (fraction of AB zeropoint flux density)
        def F_F0_ugriz(mags,key):
            return np.sinh(((np.log(10)*mags)/(-2.5) - np.log(filters[key][1])))
        
        # space for flux
        S_AB = np.zeros_like(mags)
        
        # each mags col has its own SDSS->AB conversion
        for key in filters.keys():
            for i in range(0, np.shape(mags)[1]):
                # https://www.sdss.org/dr12/algorithms/fluxcal/
                mags[:,i] = mags[:,i] - filters[key][3]
                S_AB[:,i] = 10**(-26)*3631*F_F0_ugriz(mags[:,i], key)
    
        return S_AB, ugriz_freqs
    
    # YJHK 2MASS MAGS
    elif mag_type == 'YJHK':
        # assuming no asinh->AB mag correction
        YJHK_freq = np.array([c/1031e-9, c/1248e-9, c/1631e-9, c/2201e-9])
        
        return YJHK_freq
    
    # WISE MAGS    
    elif mag_type == 'WISE':
        # assuming WISE is AB mags
        WISE_freqs = np.array([c/3.4e-6, c/4.6e-6, c/12e-6, c/22e-6])
        # flux zero points
        fw1,fw2,fw3,fw4 = 309.540,171.787,31.674,8.363
    
        zero_points = {
                       '1':fw1,
                       '2':fw2,
                       '3':fw3,
                       '4':fw4
                       }
    
        def S_WISE(mags,key):
            # Janksy flux * 10^-26 = Wm^-2Hz^-1
            return 10**(-26)*zero_points[str(key)]*10**(mags/-2.5)
        
        # space for flux
        S_AB = np.zeros_like(mags)
        
        for key in zero_points.keys():
            for i in range(0, np.shape(mags)[1]):
                S_AB[:,i] = S_WISE(mags[:,i],key)
        
        return S_AB, WISE_freqs
        
flux_DR12_ugriz, ugriz_freqs = mag_to_flux(ugriz_mags2,'ugriz') 
flux_DR12_WISE, WISE_freqs = mag_to_flux(WISE_mags,'WISE')
YJHK_freqs = mag_to_flux(0,'YJHK')

# Y,J,K,H DR12 flux, filter by SNR? 
fY, fJ = t_DR12_DR12_matches['YFLUX']-, t_DR12_DR12_matches['JFLUX']
fH, fK = t_DR12_DR12_matches['HFLUX'], t_DR12_DR12_matches['KFLUX']

fY_err, fJ_err = t_DR12_DR12_matches['YFLUX_ERR'], t_DR12_DR12_matches['JFLUX_ERR'] 
fH_err, fK_err = t_DR12_DR12_matches['HFLUX_ERR'], t_DR12_DR12_matches['KFLUX_ERR'] 

ugriz_Ang = np.array([3540, 4750, 6220, 7630, 9050])
WISE_Ang = np.array([34000, 46000, 120000, 220000])
YJHK_Ang = np.array([10310, 12480, 16310, 22010])


# EXTINCTION FOR EFFECTIVE FREQS (NOTE EXTINCTION USES WAVELENGTHS in Ang)
# Fitzpatrick & Massa (2007) function has a fixed RV of 3.1 (R&L use RV=3.1) 
# AV set as 1.
YJHK_ext = extinction.fm07(YJHK_Ang, 1.0)
ugriz_ext = extinction.fm07(ugriz_Ang, 1.0)
WISE_ext = extinction.fm07(WISE_Ang, 1.0)


    
#%%

