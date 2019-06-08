"""
author: jed homer
"""

# %% initial
import os

@@ -15,10 +11,10 @@
# os.path.abspath(os.curdir)
# os.chdir("..")

path_DR7 = "data\DR7Q.fit"
path_DR7 = "data/DR7Q.fit"
t_DR7 = Table.read(path_DR7)
path_DR7Q = "data/DR7QSOdata.fits"
t_DR7Q = Table.read(path_DR7Q)  # same table as DR7 for some reason?
t_DR7Q = Table.read(path_DR7Q) # DR7 QSOs cat.
path_DR12 = "data/DR12Q.fits"
t_DR12 = Table.read(path_DR12)
path_3XMM = "data/3XMM_DR7cat.fits"
@@ -34,10 +30,7 @@
# %% Dynamic match radius correlation

# dynamic sep. constraints: 1arcsec sym. err. for SDSS cats. and object pos errs for 3XMM
dpos1 = np.sqrt(
    np.square(np.array(t_3XMM["SC_POSERR"]))
    + np.square(np.full((499266), 0.0 * u.arcsec))
)
dpos1 = np.sqrt(np.square(np.array(t_3XMM["SC_POSERR"])) + np.square(np.full((499266), 0.0 * u.arcsec)))

coords_DR7 = SkyCoord(t_DR7["RA"] * u.deg, t_DR7["DEC"] * u.deg)
coords_DR12 = SkyCoord(t_DR12["RA"], t_DR12["DEC"])
@@ -48,25 +41,12 @@
idx_3XMM2, d2d_3XMM2, d3d_3XMM2 = coords_DR12.match_to_catalog_sky(coords_3XMM)

k = 2.0  # search radius

# coords, position errors for DR7,DR12 objects with separation less than 2" sep from 3XMM objects
cs_3XMMa, c_errs_3XMMa = (
    coords_DR7[d2d_3XMM1 < k * u.arcsec],
    dpos1[idx_3XMM1[[d2d_3XMM1 < k * u.arcsec]]],
)
cs_3XMMb, c_errs_3XMMb = (
    coords_DR12[d2d_3XMM2 < k * u.arcsec],
    dpos1[idx_3XMM2[[d2d_3XMM2 < k * u.arcsec]]],
)
cs_3XMMa, c_errs_3XMMa = (coords_DR7[d2d_3XMM1 < k * u.arcsec],dpos1[idx_3XMM1[[d2d_3XMM1 < k * u.arcsec]]])
cs_3XMMb, c_errs_3XMMb = (coords_DR12[d2d_3XMM2 < k * u.arcsec],dpos1[idx_3XMM2[[d2d_3XMM2 < k * u.arcsec]]])
# matching objects in 3XMM
cs1, c_errs1 = (
    coords_3XMM[idx_3XMM1[d2d_3XMM1 < k * u.arcsec]],
    dpos1[idx_3XMM1[[d2d_3XMM1 < k * u.arcsec]]],
)
cs2, c_errs2 = (
    coords_3XMM[idx_3XMM2[d2d_3XMM2 < k * u.arcsec]],
    dpos1[idx_3XMM2[[d2d_3XMM2 < k * u.arcsec]]],
)
cs1, c_errs1 = (coords_3XMM[idx_3XMM1[d2d_3XMM1 < k * u.arcsec]],dpos1[idx_3XMM1[[d2d_3XMM1 < k * u.arcsec]]])
cs2, c_errs2 = (coords_3XMM[idx_3XMM2[d2d_3XMM2 < k * u.arcsec]],dpos1[idx_3XMM2[[d2d_3XMM2 < k * u.arcsec]]])

XMMa_matches = []
XMMa_matches_poserr = []
@@ -77,9 +57,7 @@

    # find separations less than quadrature error on that object
    catalogmsk1 = c.separation(cs_3XMMa) < c_err * u.arcsec
    idxcatalog1 = np.where(catalogmsk1)[
        0
    ]  # indices for which 2D sep is less than dynamic error
    idxcatalog1 = np.where(catalogmsk1)[0]  # indices for which 2D sep is less than dynamic error

    XMMa_matches.append(cs_3XMMa[idxcatalog1])
    XMMa_matches_poserr.append(c_errs_3XMMa[idxcatalog1])
@@ -91,9 +69,7 @@

    # find separations less than quadrature error on that object
    catalogmsk1 = c.separation(cs_3XMMb) < c_err * u.arcsec
    idxcatalog1 = np.where(catalogmsk1)[
        0
    ]  # indices for which 2D sep is less than dynamic error
    idxcatalog1 = np.where(catalogmsk1)[0]  # indices for which 2D sep is less than dynamic error

    XMMb_matches.append(cs_3XMMb[idxcatalog1])
    XMMb_matches_poserr.append(c_errs_3XMMb[idxcatalog1])
@@ -106,18 +82,6 @@
XMMb_matches = np.concatenate(XMMb_matches)
XMMb_matches_poserr = np.concatenate([XMMb_matches_poserr])

rad = 1.5

count1 = 0
for obj in XMMa_matches_poserr:
    if obj < rad:
        count1 += 1

count2 = 0
for obj in XMMb_matches_poserr:
    if obj < rad:
        count2 += 1


DR7_matches = []
DR7_matches_poserr = []
@@ -128,9 +92,7 @@

    # find separations less than quadrature error on that object
    catalogmsk1 = c.separation(cs1) < c_err * u.arcsec
    idxcatalog1 = np.where(catalogmsk1)[
        0
    ]  # indices for which 2D sep is less than dynamic error
    idxcatalog1 = np.where(catalogmsk1)[0]  # indices for which 2D sep is less than dynamic error

    DR7_matches.append(cs1[idxcatalog1])
    DR7_matches_poserr.append(c_errs1[idxcatalog1])
@@ -142,9 +104,7 @@

    # find separations less than quadrature error on that object
    catalogmsk2 = c.separation(cs2) < c_err * u.arcsec
    idxcatalog2 = np.where(catalogmsk2)[
        0
    ]  # indices for which 2D sep is less than dynamic error
    idxcatalog2 = np.where(catalogmsk2)[0]  # indices for which 2D sep is less than dynamic error

    DR12_matches.append(cs2[idxcatalog2])
    DR12_matches_poserr.append(c_errs2[idxcatalog2])
@@ -157,6 +117,20 @@
DR12_matches = np.concatenate(DR12_matches)
DR12_matches_poserr = np.concatenate([DR12_matches_poserr])


#%% quasar counts
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
@@ -171,8 +145,21 @@
for a, b in zip(XMMa_matches, DR7_matches):
    diff.append(a.separation(b))

x = [i for i in range(0, len(XMMa_matches))]
fig, ax = plt.subplots(1)
ax.hist(np.concatenate(diff).arcsec)
# x = [i for i in range(0, len(XMMa_matches))]
# fig, ax = plt.subplots(1)
# ax.hist(np.concatenate(diff).arcsec)

print("DR7 matches:", count1)
print("DR12 matches:", count2)
print("3XMM/DR7 matches:", count3)
print("3XMM/DR12 matches:", count4)

# %%  

    # o perform our analysis,we utilized the observed continuum
    # flux density values at rest-frame 2500 Å(FUV) as  compiled 
    # by  Shen  et  al.  (2011) for  SDSS-DR7,  which  take  
    # into  account  both  the  emission  line contribution  and  
    # the  UV iron  complex. 

#%%