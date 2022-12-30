#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:47:07 2021

@author: pierre
"""

from reduction_detecteur import obj, obj_std, calstar, calstar_std, wave
import numpy as np
import scipy.signal
import time

lambdas_laser = np.array([980, 1080, 1310, 1550, 1742, 2004])[::-1]

lambdas_laser = np.array([1730.23,1545.07,1309.0,1123.71,987.72])
pixels_laser = scipy.signal.argrelmax(np.mean(wave,1), order=100)[0][1:-1]


pixs = np.arange(1024)
p = np.polyfit(pixels_laser, lambdas_laser, deg=1)
lambdas = np.poly1d(p)(pixs)


spec_th_temp = np.genfromtxt('./ADDITIONAL_DATA/pickles_uk_23.dat', delimiter=',')
spec_th_temp[:, 0] = spec_th_temp[:, 0]/10
spec_th_t2 = np.interp(lambdas, spec_th_temp[:, 0], spec_th_temp[:, 1])
spec_th = spec_th_t2/np.median(spec_th_t2)

spec_calstar =  np.sum(calstar[:, 430:450],1)

transmi = spec_calstar/spec_th
transmi/=np.median(transmi)

transmi = np.ma.masked_less_equal(transmi,0)
transmi = np.ma.masked_invalid(transmi)
transmi = np.ma.filled(transmi,1e-3)

for col in calstar.T:
    col /= transmi
for col in calstar_std.T:
    col /= transmi
    
for col in obj.T:
    col /= transmi
for col in obj_std.T:
    col /= transmi
    
mask = (obj > 120)+(obj < 0)
masked_obj = np.ma.masked_array(obj, mask)
masked_obj_std = np.ma.masked_array(obj_std, mask)
masked_calstar = np.ma.masked_array(calstar, mask)
masked_calstar_std = np.ma.masked_array(calstar_std, mask)

spec_cal_star_detransmitted =  np.sum(calstar[:, 430:450],1)

h_filt = np.genfromtxt("./ADDITIONAL_DATA/H_FILTER_2MASS.dat", delimiter='  ')
h_f = np.interp(lambdas, h_filt[:, 0]*10**3, h_filt[:, 1])
adu_h_calstar = np.sum(spec_cal_star_detransmitted*h_f)
j_filt = np.genfromtxt("./ADDITIONAL_DATA/J_FILTER_2MASS.dat", delimiter='  ')
j_f = np.interp(lambdas, j_filt[:, 0]*10**3, j_filt[:, 1])
adu_j_calstar = np.sum(spec_cal_star_detransmitted*j_f)

mag_h = 9.477
flux_h = 1.133e-13*10**-(0.4*mag_h)*10**4*0.251
mag_j = 9.819
flux_j = 3.129E-13*10**-(0.4*mag_j)*10**4*0.162

adu = np.mean([flux_h/adu_h_calstar, flux_j/adu_j_calstar]) #W.m^-2

pos = (np.arange(len(obj))-np.argmax(np.mean(masked_obj,0)))*0.01225

obj_final = masked_obj*adu
obj_final_std = masked_obj_std*adu
calstar_final = masked_calstar*adu
calstar_final_std = masked_calstar_std*adu

pos = -pos[901:50:-1]
lambdas = lambdas[873:150:-1]

obj_final = obj_final[873:150:-1, 901:50:-1]
obj_final_std = obj_final_std[873:150:-1, 901:50:-1]
calstar_final = calstar_final[873:150:-1, 901:50:-1]
calstar_final_std = calstar_final_std[873:150:-1, 901:50:-1]

# np.savetxt('./REDUCED_DATA/lam', lambdas)
# np.savetxt('./REDUCED_DATA/pos', pos)
# np.savetxt('./REDUCED_DATA/obj', obj_final)
# np.savetxt('./REDUCED_DATA/obj_std', obj_final_std)
# np.savetxt('./REDUCED_DATA/cal', calstar_final)
# np.savetxt('./REDUCED_DATA/cal_std', calstar_final_std)
