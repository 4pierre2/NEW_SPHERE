#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:07:38 2021

@author: pierre
"""

import glob
import numpy as np
from astropy.io import fits
from scipy import interpolate
import scipy.optimize
import scipy.signal
import scipy.ndimage as ndi 


#%%

"""
Fonction qui corrige de l'effet de shear:
Prend en entrée une image et une matrice de shear et donne en sortie l'image corrigée
"""

def unshear(im,mat):
    return scipy.ndimage.interpolation.affine_transform(im,mat)



"""
Fonction qui corrige de l'effet de shear:
Prend en entrée un cube d'image et une matrice de shear et donne en sortie cube d'images corrigée
"""

def unshear_cube(cube,mat):
    new_cube=np.copy(cube)
    i=0
    for im in cube:
        new_cube[i]=unshear(im,mat)
        i=i+1
    return new_cube


#%%


files = sorted(glob.glob('/home/pierre/Documents/2021/NEW_SPHERE/RAW_DATA/*.fits'))


obj_hdu = []
sky_hdu = []
calstar_hdu = []
dark_hdu = []
flat_hdu = []
wave_hdu = []
acq_hdu = []


for f in files:
    hdu = fits.open(f)
    header = hdu[0].header
    if header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and header['HIERARCH ESO INS1 OPTI1 ID'] == 'GRIS_MRS' and '1068' in header['OBJECT']:
        obj_hdu.append(hdu)
    if header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and header['HIERARCH ESO INS1 OPTI1 ID'] == 'GRIS_MRS' and '004134' in header['OBJECT']:
        calstar_hdu.append(hdu)
    if 'SKY' in header['OBJECT']:
        sky_hdu.append(hdu)
    if 'DARK' in header['OBJECT']:
        dark_hdu.append(hdu)
    if 'FLAT' in header['OBJECT']:
        flat_hdu.append(hdu)
    if 'WAVE' in header['OBJECT']:
        wave_hdu.append(hdu)
    if header['HIERARCH ESO DPR CATG'] == 'ACQUISITION':
        acq_hdu.append(hdu)
    # if '00413' in header['OBJECT'] and header['HIERARCH ESO DPR CATG'] == 'ACQUISITION':
    #     acq_hdu.append(hdu)
        
        
      
        
dark_ims = []
dark_stds = []
dark_dits = []   
for hdu in dark_hdu:
    header = hdu[0].header
    dit =  float(header['HIERARCH ESO DET SEQ1 DIT'])
    ndit =  float(header['HIERARCH ESO DET NDIT'])
    cube = hdu[0].data
    dark_im = np.mean(cube, 0)
    dark_std = np.std(cube, 0)  
    dark_ims.append(dark_im)
    dark_stds.append(dark_std)
    dark_dits.append(dit)
    if dit == 64:
        dark64 = np.mean(cube, 0)
    if dit == 6:
        dark6 = np.mean(cube, 0)
    
        
flat_ims = []
flat_stds = []
flat_dits = []        
for hdu in flat_hdu:
    header = hdu[0].header
    dit =  float(header['HIERARCH ESO DET SEQ1 DIT'])
    ndit =  float(header['HIERARCH ESO DET NDIT'])
    cube = hdu[0].data
    flat_im = np.mean(cube, 0)
    flat_std = np.std(cube, 0)
    if ndit == 40:
        flat_ims.append(flat_im)
        flat_stds.append(flat_std)
        flat_dits.append(dit)
        
flat_ims = np.array(flat_ims)

simple_flat = flat_ims[-1]/np.median(flat_ims[-1])

mask1 = simple_flat < 1.1
mask2 = simple_flat > 0.9
mask = mask1*mask2

masked_flat = np.ma.masked_array(simple_flat, 1-mask)


def g(x, a, b, std, x0):
    return a*np.exp(-(x-x0)**2/std)+b


obj_ims = []
obj_dits = []        
for hdu in [obj_hdu[0]]:
    header = hdu[0].header
    dit =  float(header['HIERARCH ESO DET SEQ1 DIT'])
    ndit =  float(header['HIERARCH ESO DET NDIT'])
    cube = hdu[0].data-dark64
    for im0 in cube:
        im = im0[175:180, 420:460]
        im = im/np.max(np.mean(im,0))
        p, c = scipy.optimize.curve_fit(g, range(len(np.mean(im,0))), np.mean(im,0), p0=[1,0,5,len(im[0])/2])
        if p[2]<22:
            obj_ims.append(im0)
            obj_dits.append(dit)
     
calstar_ims = []
calstar_dits = []        
for hdu in calstar_hdu:
    header = hdu[0].header
    dit =  float(header['HIERARCH ESO DET SEQ1 DIT'])
    ndit =  float(header['HIERARCH ESO DET NDIT'])
    cube = hdu[0].data-dark6
    for im0 in cube:
        calstar_ims.append(im0)
        calstar_dits.append(dit)
        
        
wave_ims = []
for hdu in wave_hdu:
    header = hdu[0].header
    cube = hdu[0].data
    for im0 in cube:
        wave_ims.append(im0)
        
for k in range(len(obj_ims)):
    obj_ims[k] = obj_ims[k]/masked_flat/obj_dits[k]
for k in range(len(calstar_ims)):
    calstar_ims[k] = calstar_ims[k]/masked_flat/calstar_dits[k]
for k in range(len(wave_ims)):
    wave_ims[k] = wave_ims[k]/masked_flat
     
acq = acq_hdu[0][0].data[0]/masked_flat
acq_l = acq*2
acq_l[:520]=0
acq_l[540:]=0
acq_l[:,:460]=0
acq_l[:,480:]=0
left_barycenter = np.array(ndi.center_of_mass(acq_l))
acq_r = acq*2
acq_r[:510]=0
acq_r[530:]=0
acq_r[:,:1490]=0
acq_r[:,1510:]=0
right_barycenter = np.array(ndi.center_of_mass(acq_r))

shift = right_barycenter-left_barycenter
shift = np.array([-11, 3])
obj_ims_realigned = []

for k in range(len(obj_ims)):
    im_left = obj_ims[k][:,:1024]*1.
    im_right = np.roll(obj_ims[k][:,1024:]*1., -np.rint(shift).astype('int'), (0, 1))
    obj_ims_realigned.append(im_left)
    obj_ims_realigned.append(im_right)
    

obj_ims_realigned = np.ma.masked_array(obj_ims_realigned)
obj_ims_realigned = np.ma.masked_outside(obj_ims_realigned,0,120)
obj = obj_ims_realigned.mean(0)


for im in obj_ims_realigned:
    smoothed_im = scipy.signal.medfilt(im, kernel_size=3)
    dif = im-smoothed_im
    im.mask += dif>10


calstar_ims_realigned = []

for k in range(len(calstar_ims)):
    im_left = calstar_ims[k][:,:1024]*1.
    im_right = np.roll(calstar_ims[k][:,1024:]*1., -np.rint(shift).astype('int'), (0, 1))
    calstar_ims_realigned.append(im_left)
    calstar_ims_realigned.append(im_right)
    

calstar_ims_realigned = np.ma.masked_array(calstar_ims_realigned)
calstar = calstar_ims_realigned.mean(0)


wave_ims_realigned = []

for k in range(len(wave_ims)):
    im_left = wave_ims[k][:,:1024]*1.
    im_right = np.roll(wave_ims[k][:,1024:]*1., -np.rint(shift).astype('int'), (0, 1))
    wave_ims_realigned.append(im_left)
    wave_ims_realigned.append(im_right)
    

wave_ims_realigned = np.ma.masked_array(wave_ims_realigned)
wave = wave_ims_realigned.mean(0)


matrix_shear=[[1.,0.,0],[-0.006553,1.,0],[0,0,1]]


obj = unshear(obj, matrix_shear)
calstar = unshear(calstar, matrix_shear)
wave = unshear(wave, matrix_shear)
