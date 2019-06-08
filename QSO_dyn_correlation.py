# %% initial
import os
import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.coordinates import match_coordinates_sky

#os.path.abspath(os.curdir)
#os.chdir("..")

path_DR7 = 'data\DR7Q.fit'
t_DR7 = Table.read(path_DR7)
path_DR7Q = 'data/DR7QSOdata.fits'
t_DR7Q = Table.read(path_DR7Q)  # same table as DR7 for some reason?
path_DR12 = 'data/DR12Q.fits'
t_DR12 = Table.read(path_DR12)
path_3XMM = 'data/3XMM_DR7cat.fits'
t_3XMM = Table.read(path_3XMM)

# REMOVE BAL_FLAG != 0 quasars () in SDSS DR7 AND radio loud quasars in SDSS DR7
t_DR7Q = t_DR7Q[(t_DR7Q['BAL_FLAG'] == 0) & (t_DR7Q['R_6CM_2500A'] < 10)]
t_DR7 = t_DR7Q

# DR12 BAL removal (removes ~ 10%)
t_DR12 = t_DR12[(t_DR12['BAL_FLAG_VI'] == 0)]

# %% Dynamic match radius correlation

# dynamic sep. constraints: 1arcsec sym. err. for SDSS cats. and object pos errs for 3XMM
dpos1 = np.sqrt(np.square(np.array(t_3XMM['SC_POSERR'])) + np.square(np.full((499266), 0.0*u.arcsec)))

coords_DR7 = SkyCoord(t_DR7['RA']*u.deg, t_DR7['DEC']*u.deg)
coords_DR12 = SkyCoord(t_DR12['RA'], t_DR12['DEC'])
coords_3XMM = SkyCoord(t_3XMM['SC_RA'], t_3XMM['SC_DEC'])

# catalog matching DR7,DR12 to 3XMM (CLOSEST MATCH )
idx_3XMM1, d2d_3XMM1, d3d_3XMM1 = coords_DR7.match_to_catalog_sky(coords_3XMM)
idx_3XMM2, d2d_3XMM2, d3d_3XMM2 = coords_DR12.match_to_catalog_sky(coords_3XMM)

k = 2.0 # search radius

# coords, position errors for DR7,DR12 objects with separation less than 2" sep from 3XMM objects
cs_3XMMa, c_errs_3XMMa = coords_DR7[d2d_3XMM1 < k*u.arcsec], dpos1[idx_3XMM1[[d2d_3XMM1 < k*u.arcsec]]]
cs_3XMMb, c_errs_3XMMb = coords_DR12[d2d_3XMM2 < k*u.arcsec], dpos1[idx_3XMM2[[d2d_3XMM2 < k*u.arcsec]]]
# matching objects in 3XMM 
cs1, c_errs1 = coords_3XMM[idx_3XMM1[d2d_3XMM1 < k*u.arcsec]], dpos1[idx_3XMM1[[d2d_3XMM1 < k*u.arcsec]]]
cs2, c_errs2 = coords_3XMM[idx_3XMM2[d2d_3XMM2 < k*u.arcsec]], dpos1[idx_3XMM2[[d2d_3XMM2 < k*u.arcsec]]]

XMMa_matches = []
XMMa_matches_poserr = []
XMMb_matches = []
XMMb_matches_poserr = []

DR7_matches = []
DR7_matches_poserr = []
DR12_matches = []
DR12_matches_poserr = []

for c, c_err in np.column_stack((cs_3XMMa, c_errs_3XMMa)):
    
    # find separations less than quadrature error on that object
    catalogmsk1 = c.separation(cs_3XMMa) < c_err*u.arcsec
    idxcatalog1 = np.where(catalogmsk1)[0]  # indices for which 2D sep is less than dynamic error
    
    XMMa_matches.append(cs_3XMMa[idxcatalog1])
    XMMa_matches_poserr.append(c_errs_3XMMa[idxcatalog1])
    
    print(cs_3XMMa[idxcatalog1], c_errs_3XMMa[idxcatalog1])
    print(len(XMMa_matches)) 


for c, c_err in np.column_stack((cs_3XMMb, c_errs_3XMMb)):
    
    # find separations less than quadrature error on that object
    catalogmsk1 = c.separation(cs_3XMMb) < c_err*u.arcsec
    idxcatalog1 = np.where(catalogmsk1)[0]  # indices for which 2D sep is less than dynamic error
    
    XMMb_matches.append(cs_3XMMb[idxcatalog1])
    XMMb_matches_poserr.append(c_errs_3XMMb[idxcatalog1])
    
    print(cs_3XMMb[idxcatalog1], c_errs_3XMMb[idxcatalog1])
    print(len(XMMb_matches)) 

for c, c_err in np.column_stack((cs1, c_errs1)):
    
    # find separations less than quadrature error on that object
    catalogmsk1 = c.separation(cs1) < c_err*u.arcsec
    idxcatalog1 = np.where(catalogmsk1)[0]  # indices for which 2D sep is less than dynamic error
    
    DR7_matches.append(cs1[idxcatalog1])
    DR7_matches_poserr.append(c_errs1[idxcatalog1])
    
    print(cs1[idxcatalog1], c_errs1[idxcatalog1])
    print(len(DR7_matches))
    
for c, c_err in np.column_stack((cs2, c_errs2)):

    # find separations less than quadrature error on that object
    catalogmsk2 = c.separation(cs2) < c_err*u.arcsec
    idxcatalog2 = np.where(catalogmsk2)[0]  # indices for which 2D sep is less than dynamic error
    
    DR12_matches.append(cs2[idxcatalog2])
    DR12_matches_poserr.append(c_errs2[idxcatalog2])
    
    print(cs2[idxcatalog2], c_errs2[idxcatalog2])
    print(len(DR12_matches))

XMMa_matches = np.concatenate(XMMa_matches)
XMMa_matches_poserr = np.concatenate([XMMa_matches_poserr])
XMMb_matches = np.concatenate(XMMb_matches)
XMMb_matches_poserr = np.concatenate([XMMb_matches_poserr])

DR7_matches = np.concatenate(DR7_matches)
DR7_matches_poserr = np.concatenate([DR7_matches_poserr])
DR12_matches = np.concatenate(DR12_matches)
DR12_matches_poserr = np.concatenate([DR12_matches_poserr])

rad = 1.5

count1 = 0
for obj in XMMa_matches_poserr:
    if obj < rad:
        count1 += 1

count2 = 0
for obj in XMMb_matches_poserr:
    if obj < rad:
        count2 += 1

count3 = 0
for obj in DR7_matches_poserr:
    if obj < rad:
        count3 += 1

count4 = 0
for obj in DR12_matches_poserr:
    if obj < rad:
        count4 += 1

diff = []
for a,b in zip(XMMa_matches,DR7_matches):
    diff.append(a.separation(b))

print(count1,count2,count3,count4)