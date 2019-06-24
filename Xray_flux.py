# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:50:49 2019

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

"""
    integrate power law for Xrays for hard and soft x rays to find flux in either
    band.
    
    use this as a function of energy to construct function of flux, extrapolate
    this to observed (or rest frame, depending) Xray energy at (1+z)*2keV (or 
    2keV).
    
    fit the new function at the soft and hard bands to get the function to extrapolate.
"""


# %% redshifts

# redshifts 'z' 
z_DR7, z_DR12 = t_DR7_DR7_matches['REDSHIFT'], t_DR12_DR12_matches['Z_CIV']

# remove bad redshifts
z_DR12[z_DR12 < 0] = np.nan
z_DR12 = np.ma.array(z_DR12, mask=np.isnan(z_DR12))


# %% XRAY OBSERVED FLUXES

# FLUXES ARE IN ERGS/CM^2

#DR7,DR12 objs with 3XMM xray measures
FX_soft_DR7 = t_3XMM_DR7_matches['SC_EP_2_FLUX'] + t_3XMM_DR7_matches['SC_EP_3_FLUX']
FX_soft_err_DR7 = t_3XMM_DR7_matches['SC_EP_2_FLUX_ERR'] + t_3XMM_DR7_matches['SC_EP_3_FLUX_ERR']

FX_hard_DR7 = t_3XMM_DR7_matches['SC_EP_4_FLUX'] + t_3XMM_DR7_matches['SC_EP_5_FLUX']
FX_hard_err_DR7 = t_3XMM_DR7_matches['SC_EP_4_FLUX_ERR'] + t_3XMM_DR7_matches['SC_EP_5_FLUX_ERR']

FX_soft_DR12 = t_3XMM_DR12_matches['SC_EP_2_FLUX'] + t_3XMM_DR12_matches['SC_EP_3_FLUX']
FX_soft_err_DR12 = t_3XMM_DR12_matches['SC_EP_2_FLUX_ERR'] + t_3XMM_DR12_matches['SC_EP_3_FLUX_ERR']

FX_hard_DR12 = t_3XMM_DR12_matches['SC_EP_4_FLUX'] + t_3XMM_DR12_matches['SC_EP_5_FLUX']
FX_hard_err_DR12 = t_3XMM_DR12_matches['SC_EP_4_FLUX_ERR'] + t_3XMM_DR12_matches['SC_EP_5_FLUX_ERR']

# %% FX

gamma = 1.7
E1 = 0.5
E2 = 2.0
ES = 1.05
keV_J = 1.6e-16


def F_ES(flux):
    # calculate integrated function vals for fluxes (fluxes into keV from J!!!!)
    # flux out of table is in ergs
    
    F_ES = np.zeros_like(flux)
    for i in range(0,len(flux)):
        F_ES[i] = ((flux[i]*keV_J)*(gamma+1)*(ES)**gamma)/(np.abs(E2**(gamma+1) - E1**(gamma+1)))
    
    return F_ES/keV_J

# soft & hard X-ray fluxes frm power law integral
FXS_DR7, FXH_DR7 = F_ES(FX_soft_DR7), F_ES(FX_hard_DR7)
FXS_DR12, FXH_DR12 = F_ES(FX_soft_DR12), F_ES(FX_hard_DR12)

def F_ES_fit_eq(flux_int1, flux_int2, E):
    
    params = np.zeros((len(flux_int1),2))
    
    f_2keV = np.zeros((len(flux_int1),))
    
    bands = [1.25, 7.0] # band midpoints 0.5-2keV, 2-12keV
    
    for i in range(0, len(flux_int1)):
        
        params[i,0] = np.polyfit(
                bands,
                [flux_int1[i], flux_int2[i]],
                1
                )[0]
        
        params[i,1] = np.polyfit(
                bands,
                [flux_int1[i], flux_int2[i]],
                1
                )[1]
    
        
        f_2keV[i] = params[i,0]*E[i] + params[i,1]
    
        
        print(100*i/len(flux_int1)//1,'%')
        
        
    
    return f_2keV


# fluxes at 2keV from extrapolation
E_rest1, E_rest2 = 2*(1+z_DR7), 2*(1+z_DR12)

DR7_2keV = F_ES_fit_eq(FXS_DR7, FXH_DR7, E_rest1)
DR12_2keV = F_ES_fit_eq(FXS_DR12, FXH_DR12, E_rest2)

# %% LX

# luminosity distance
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
DL_DR7, DL_DR12 = cosmo.luminosity_distance(z_DR7), cosmo.luminosity_distance(z_DR12)
DL_DR7, DL_DR12 = DL_DR7.to(u.cm), DL_DR12.to(u.cm)

L_X_DR7 = (10**(DR7_2keV)/(1+z_DR7))*(4*np.pi*np.square(DL_DR7.value)) # ergs/J conversion factor
L_X_DR12 = (10**(DR12_2keV)/(1+z_DR12))*(4*np.pi*np.square(DL_DR12.value)) # ergs/J conversion factor
