#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:47:59 2018

@author: pvermot
"""

from spec_obj import Spec_Obj
import numpy as np
import sys
import os
import numpy.ma as ma
from copy import copy, deepcopy
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#%%

pos = np.loadtxt('./REDUCED_DATA/pos')
lam = np.loadtxt('./REDUCED_DATA/lam')
obj = np.loadtxt('./REDUCED_DATA/obj')
obj_std = np.loadtxt('./REDUCED_DATA/obj_std')
cal = np.loadtxt('./REDUCED_DATA/cal')
cal_std = np.loadtxt('./REDUCED_DATA/cal_std')

ngc1068 = Spec_Obj(obj, obj_std, lam, pos)
bd00413 = Spec_Obj(cal, cal_std, lam, pos)

#%%

def make_dir(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def get_mask_lam(good_lams):
    good_lams.append(10000)
    lams = [0]+good_lams
    mask_lams = []
    for k in range(int(len(lams)/2)):
        mask_lams.append([lams[2*k],lams[2*k+1], 'default', 'default'])
    return mask_lams

lams= [969, 974, 1052, 1057, 1170, 1175, 1225, 1230, 1300, 1305, 1437, 1442, 1520, 1525, 1650, 1655, 1687, 1692, 1740, 1745, 1780, 1785]

lams= [969, 974, 1052, 1057, 1170, 1175, 1225, 1230, 1310, 1315, 1437, 1442, 1520, 1525, 1650, 1655, 1687, 1692, 1740, 1745, 1780, 1785]

mask_lams = get_mask_lam(lams)
        
for ls in mask_lams:
    ngc1068.mask_region(lam_0=ls[0],lam_1=ls[1], pos_0=ls[2], pos_1=ls[3])
    
deg = 3
ngc1068.def_cont(ngc1068.make_poly_cont_from_not_masked_regions(deg=deg)[0], std=ngc1068.make_poly_cont_from_not_masked_regions(deg=deg)[1])
ngc1068.def_raies_from_cont(ngc1068.make_poly_cont_from_not_masked_regions(deg=deg)[0], std=ngc1068.make_poly_cont_from_not_masked_regions(deg=deg)[1])

cont=ngc1068.make_cont_obj()
lines=ngc1068.make_lines_obj()
ngc1068.reset_mask()


#%%


def interpolate_masked_array(array, method='linear'):
    array = array
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    
    GD1 = scipy.interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                 method=method)
    return GD1

im = ngc1068.obj.data
smoothed_im = scipy.signal.medfilt(im, kernel_size=3)
dif = abs(im-smoothed_im)
new_mask = (dif+1e-20)>3e-19
ngc1068.obj.mask += new_mask
ngc1068.obj = interpolate_masked_array(ngc1068.obj)
lines.obj.mask += new_mask
lines.obj = interpolate_masked_array(lines.obj)


o = deepcopy(ngc1068)

im = deepcopy(o.obj)

pn0a = ngc1068.ptx(-0.06)
pn0b = ngc1068.ptx(0.15)
pn1a = ngc1068.ptx(0.2)
pn1b = ngc1068.ptx(0.4)
pn2a = ngc1068.ptx(0.59)
pn2b = ngc1068.ptx(0.8)

pn0a = ngc1068.ptx(-0.06)
pn0b = ngc1068.ptx(0.15)
pn1a = ngc1068.ptx(0.25)
pn1b = ngc1068.ptx(0.35)
pn2a = ngc1068.ptx(0.65)
pn2b = ngc1068.ptx(0.76)

print(pn0b-pn0a, pn1b-pn1a, pn2b-pn2a)

mask_nod0 = np.zeros(np.shape(im), dtype='bool')
mask_nod0[:, pn0a:pn0b+1] = True
mask_nod1 = np.zeros(np.shape(im), dtype='bool')
mask_nod1[:, pn1a:pn1b+1] = True
mask_nod2 = np.zeros(np.shape(im), dtype='bool')
mask_nod2[:, pn2a:pn2b+1] = True

interp_for_nod0 = interpolate_masked_array(np.ma.masked_array(im, mask=mask_nod0))
interp_for_nod1 = interpolate_masked_array(np.ma.masked_array(im, mask=mask_nod1))
interp_for_nod2 = interpolate_masked_array(np.ma.masked_array(im, mask=mask_nod2))

nod0=deepcopy(o)
nod1=deepcopy(o)
nod2=deepcopy(o)

nod0.obj = nod0.obj-interp_for_nod0
nod0.std[~mask_nod0] = 0
nod1.obj = nod1.obj-interp_for_nod1
nod1.std[~mask_nod1] = 0
nod2.obj = nod2.obj-interp_for_nod2
nod2.std[~mask_nod2] = 0


#%%

# phei = lines.plot_prof(1077,1090)
# plt.fill_between([-0.015,0.085],[np.max(phei+err_phei),np.max(phei+err_phei)])
# plt.fill_between([0.25,0.35],[np.max(phei+err_phei),np.max(phei+err_phei)])
# plt.fill_between([0.65,0.75],[np.max(phei+err_phei),np.max(phei+err_phei)])

p, psix, err_psix = lines.make_prof(1424,1437)
plt.fill_between([-0.065,0.1335],[np.max(psix+err_psix),np.max(psix+err_psix)], alpha=0.2, color='blue', label='Nodule 1')
plt.fill_between([0.18,0.38],[np.max(psix+err_psix),np.max(psix+err_psix)], alpha=0.2, color='green', label='Nodule 2')
plt.fill_between([0.60,0.8],[np.max(psix+err_psix),np.max(psix+err_psix)], alpha=0.2, color='red', label='Nodule 3')
lines.plot_prof(1424,1437, color_err='k')
plt.xlim(-0.25,1)
plt.ylim(0,1.75e-16)
plt.title('[Si X] spatial distribution')
plt.legend()
plt.savefig('./plots/six_profile.pdf')
plt.savefig('./plots/six_profile.png')

plt.figure()
p, phei, err_phei = lines.make_prof(1077,1090)
plt.fill_between([-0.065,0.1335],[1.2e-15,1.2e-15], alpha=0.2, color='blue', label='Nodule 1')
plt.fill_between([0.18,0.38],[1.2e-15,1.2e-15], alpha=0.2, color='green', label='Nodule 2')
plt.fill_between([0.60,0.8],[1.2e-15,1.2e-15], alpha=0.2, color='red', label='Nodule 3')
lines.plot_prof(1077,1090, color_err='k')
plt.xlim(-0.25,1)
plt.ylim(0,1.2e-15)
plt.title('He I spatial distribution')
plt.legend()
plt.savefig('./plots/hei_profile.pdf')
plt.savefig('./plots/hei_profile.png')


#%%

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10,6))
# Remove horizontal space between axes

# Plot each graph, and manually set the y tick values

plt.axes(axs[0])
lines.plot_spec(-0.065,0.1335, step=True, color_err='blue', title='', bins=2)
p, spec_nod1, err_spec_nod1 = lines.make_spec(-0.065,0.1335)
plt.axes(axs[1])
lines.plot_spec(0.18,0.38, step=True, color_err='green', title='', bins=2)
p, spec_nod2, err_spec_nod2 = lines.make_spec(0.18,0.38)
plt.axes(axs[2])
lines.plot_spec(0.6,0.8, step=True, color_err='red', title='', bins=2)
p, spec_nod3, err_spec_nod3 = lines.make_spec(0.6,0.8)
plt.tight_layout()
fig.subplots_adjust(hspace=0)
plt.savefig('./plots/spec_nodules.pdf')
# phei = lines.plot_prof(10,1090)
# plt.fill_between([-0.015,0.085],[np.max(phei),np.max(phei)])

# l, nod0, err0 = ngc1068.

# interp_for_nod12 = interpolate_masked_array(np.ma.masked_array(im, mask=mask_nod2))

#%%
def db_gauss_hei_pag(x, a1, b, c, a0):
    return a0*np.exp(-c*(x-b*1083.0/1093.8)**2)+a1*np.exp(-c*(x-b)**2)
def gauss(x, a, b, c):
    return a*np.exp(-c*(x-b)**2)


l = [1070,1110]
x0 = lines.ltx(l[0])
x1 = lines.ltx(l[1])
p1, cov1 = curve_fit(db_gauss_hei_pag, lines.lam[x0:x1], spec_nod1[x0:x1], sigma=err_spec_nod1[x0:x1], p0=[np.max(spec_nod1[x0:x1])/20, 1093.8, 1, np.max(spec_nod1[x0:x1])])
flux_pagam_1 = p1[0]*(np.pi/p1[2])**0.5/1e3
err_pagam_1 = np.sqrt(np.diag(cov1))
plt.plot(lines.lam[x0:x1], spec_nod1[x0:x1])
plt.plot(lines.lam[x0:x1], db_gauss_hei_pag(lines.lam[x0:x1], *p1))

l = [1269,1291]
x0 = lines.ltx(l[0])
x1 = lines.ltx(l[1])
p1, cov1 = curve_fit(gauss, lines.lam[x0:x1], spec_nod1[x0:x1], sigma=err_spec_nod1[x0:x1], p0=[np.max(spec_nod1[x0:x1]), 1282, 1])
flux_pabet_1 = p1[0]*(np.pi/p1[2])**0.5/1e3
err_pabet_1 = np.sqrt(np.diag(cov1))
plt.plot(lines.lam[x0:x1], spec_nod1[x0:x1])
plt.plot(lines.lam[x0:x1], gauss(lines.lam[x0:x1], *p1))

l = [1001,1019]
x0 = lines.ltx(l[0])
x1 = lines.ltx(l[1])
p1, cov1 = curve_fit(gauss, lines.lam[x0:x1], spec_nod1[x0:x1], sigma=err_spec_nod1[x0:x1], p0=[np.max(spec_nod1[x0:x1]), 1010, 1])
flux_heii_1 = p1[0]*(np.pi/p1[2])**0.5/1e3
err_pabet_1 = np.sqrt(np.diag(cov1))
plt.plot(lines.lam[x0:x1], spec_nod1[x0:x1])
plt.plot(lines.lam[x0:x1], gauss(lines.lam[x0:x1], *p1))



l = [1070,1110]
x0 = lines.ltx(l[0])
x1 = lines.ltx(l[1])
p2, cov2 = curve_fit(db_gauss_hei_pag, lines.lam[x0:x1], spec_nod2[x0:x1], sigma=err_spec_nod2[x0:x1], p0=[np.max(spec_nod2[x0:x1])/20, 1093.8, 1, np.max(spec_nod2[x0:x1])])
flux_pagam_2 = p2[0]*(np.pi/p2[2])**0.5/1e3
err_pagam_2 = np.sqrt(np.diag(cov2))
plt.plot(lines.lam[x0:x1], spec_nod2[x0:x1])
plt.plot(lines.lam[x0:x1], db_gauss_hei_pag(lines.lam[x0:x1], *p2))


l = [1269,1291]
x0 = lines.ltx(l[0])
x1 = lines.ltx(l[1])
p2, cov2 = curve_fit(gauss, lines.lam[x0:x1], spec_nod2[x0:x1], sigma=err_spec_nod2[x0:x1], p0=[np.max(spec_nod2[x0:x1]), 1282, 1])
flux_pabet_2 = p2[0]*(np.pi/p2[2])**0.5/1e3
err_pabet_2 = np.sqrt(np.diag(cov2))


l = [1001,1019]
x0 = lines.ltx(l[0])
x1 = lines.ltx(l[1])
p2, cov2 = curve_fit(gauss, lines.lam[x0:x1], spec_nod2[x0:x1], sigma=err_spec_nod2[x0:x1], p0=[np.max(spec_nod1[x0:x1]), 1010, 1])
flux_heii_1 = p1[0]*(np.pi/p1[2])**0.5/1e3
err_pabet_1 = np.sqrt(np.diag(cov1))
plt.plot(lines.lam[x0:x1], spec_nod1[x0:x1])
plt.plot(lines.lam[x0:x1], gauss(lines.lam[x0:x1], *p1))

l = [1070,1110]
x0 = lines.ltx(l[0])
x1 = lines.ltx(l[1])
p3, cov3 = curve_fit(db_gauss_hei_pag, lines.lam[x0:x1], spec_nod3[x0:x1], sigma=err_spec_nod3[x0:x1], p0=[np.max(spec_nod3[x0:x1])/20, 1093.8, 1, np.max(spec_nod3[x0:x1])])
flux_pagam_3 = p3[0]*(np.pi/p3[2])**0.5/1e3
err_pagam_3 = np.sqrt(np.diag(cov3))
plt.plot(lines.lam[x0:x1], spec_nod3[x0:x1])
plt.plot(lines.lam[x0:x1], db_gauss_hei_pag(lines.lam[x0:x1], *p3))



l = [1269,1291]
x0 = lines.ltx(l[0])
x1 = lines.ltx(l[1])
p3, cov3 = curve_fit(gauss, lines.lam[x0:x1], spec_nod3[x0:x1], sigma=err_spec_nod3[x0:x1], p0=[np.max(spec_nod3[x0:x1]), 1282, 1])
flux_pabet_3 = p3[0]*(np.pi/p3[2])**0.5/1e3
err_pabet_3 = np.sqrt(np.diag(cov3))



# file = open("extinction_lines.txt","w")
# file.write("\\"+'begin{table}\r')
# file.write("\\"+'centering\r')
# file.write("\\caption{\label{table_lines} Extinction measurements}\r")
# file.write("\\begin{tabular}{c|c|c|c}\r")
# file.write("Position & Flux $Pa_{\gamma}$ & $Flux Pa_{\beta}$ ")
# file.write("Nodule 1 & $"+str($ & $Flux Pa_{\beta}$ ")



#%%
lams = [[975,996], [1001,1019], [1012, 1048], [1070,1095], [1153, 1175], [1176, 1998], [1243, 1261], [1269,1291], [1418, 1437]]
ls=np.mean(lams,1)
ls_ref=[985.0, 1012.2, 1032.0, 1083.0, 1188.6, 1252.0, 1282.0, 1430.0, 1643.6]
names = ['[C I]', 'He II', '[S II]', 'He I', '[?]', '[P II]', '[S IX]', r'$Pa_{\beta}$', '[Si X]', '[Fe II]']

def totex(x, dx):
    mini = np.min([x,dx])
    power = int(np.log10(mini))
    return r'$'+'{:.1f}'.format(x/10**power)+'\\'+'pm'+'{:.1f}'.format(dx/10**power)+'\\times 10^{'+str(power)+'}$'
#%%
def gauss(x, a, b, c):
    return a*np.exp(-c*(x-b)**2)

def db_gauss_hei_pag(x, a1, b, c, a0):
    return a0*np.exp(-c*(x-b*1083.0/1093.8)**2)+a1*np.exp(-c*(x-b)**2)


file = open("table_lines.txt","w")
file.write("\\"+'begin{table}\r')
file.write("\\"+'centering\r')
file.write("\\caption{\label{table_lines} Emission lines summary}\r")
file.write("\\begin{tabular}{c|c|c|c}\r")
file.write("Lines & Rest wavelength & $F_{int}$ & $F_{g} & $\lambda_{g}\r")
for k in range(len(lams)):
    l = lams[k]
    x0 = lines.ltx(l[0])
    x1 = lines.ltx(l[1])
    l0_true = lines.lam[0]
    l1_true = lines.lam[1]
    flux_int_1 = np.sum(spec_nod1[x0:x1])/(l1_true-l0_true)/1e3
    err_int_1 = np.sum(err_spec_nod1[x0:x1])/(l1_true-l0_true)/1e3
    flux_int_2 = np.sum(spec_nod2[x0:x1])/(l1_true-l0_true)/1e3
    err_int_2 = np.sum(err_spec_nod2[x0:x1])/(l1_true-l0_true)/1e3
    flux_int_3 = np.sum(spec_nod3[x0:x1])/(l1_true-l0_true)/1e3
    err_int_3 = np.sum(err_spec_nod3[x0:x1])/(l1_true-l0_true)/1e3
    
    if names[k] != 'He I':
        p1, cov1 = curve_fit(gauss, lines.lam[x0:x1], spec_nod1[x0:x1], sigma=err_spec_nod1[x0:x1], p0=[np.max(spec_nod1[x0:x1]), ls[k], 1])
        flux_g_1 = p1[0]*(np.pi/p1[2])**0.5/1e3
        err1 = np.sqrt(np.diag(cov1))
        err_g_1 = ((err1[0]*(np.pi/p1[2])**0.5)**2+(p1[0]*err1[2]*np.pi**0.5*1.5*p1[2]**(-1.5))**2)**0.5/1e3
        p2, cov2 = curve_fit(gauss, lines.lam[x0:x1], spec_nod2[x0:x1], p0=[np.max(spec_nod2[x0:x1]), ls[k], 1])
        flux_g_2 = p2[0]*(np.pi/p2[2])**0.5/1e3
        err2 = np.sqrt(np.diag(cov2))
        err_g_2 = ((err2[0]*(np.pi/p2[2])**0.5)**2+(p2[0]*err2[2]*np.pi**0.5*1.5*p2[2]**(-1.5))**2)**0.5/1e3
        p3, cov3 = curve_fit(gauss, lines.lam[x0:x1], spec_nod3[x0:x1], p0=[np.max(spec_nod3[x0:x1]), ls[k], 1])
        flux_g_3 = p3[0]*(np.pi/p2[2])**0.5/1e3
        err3 = np.sqrt(np.diag(cov3))
        err_g_3 = ((err3[0]*(np.pi/p3[2])**0.5)**2+(p3[0]*err3[2]*np.pi**0.5*1.5*p3[2]**(-1.5))**2)**0.5/1e3
        # file.write(names[k]+' & '+'{:.1f}'.format(ls_ref[k])+'$\ nm$ & '+totex(flux_g_1, err_g_1)+'$\ W.m^{-2}$ & '+
    # else: 
    #     p2, cov2 = curve_fit(db_gauss_hei_pag, lines.lam[x0:x1], spec_nod2[x0:x1], sigma=err_spec_nod2[x0:x1], p0=[np.max(spec_nod2[x0:x1])/20, 1093.8, 1, np.max(spec_nod2[x0:x1])])
    #     flux_pagam_2 = p2[0]*(np.pi/p2[2])**0.5/1e3
    #     err_pagam_2 = np.sqrt(np.diag(cov2))
    #     plt.plot(lines.lam[x0:x1], spec_nod2[x0:x1])
    #     plt.plot(lines.lam[x0:x1], db_gauss_hei_pag(lines.lam[x0:x1], *p2))


    # print(names[k], flux_int_1, err_int_1, flux_int_2, err_int_2, flux_int_3, err_int_3)
    print(names[k],p1, err1, p2, err2, p3, err3)
    # file.write(names[k]+' & '+'{:.1f}'.format(ls_ref[k])+' & '+'{:.1f}'.format(flux
file.write('\end{tabular}')
file.close()
#%%
"""
#%%
plt.figure(30)
cont.plot_prof(1730,1770, legend='1750 nm')
cont.plot_prof(980,1020, color='b', legend='1000 nm')
plt.title('')

plt.figure(31)
cont.plot_prof(1730,1770, lognorm='y', legend='1750 nm')
cont.plot_prof(980,1020, lognorm='y', color='b', legend='1000 nm')
plt.title('')
#%%

ngc.reset_mask()
#%%

lines.plot_spec(1,2)

#%%
scipy.ndimage.shift

"""
"""
a_983=lines.make_line_obj(975,998, wvl=983.82, dble=990.6, name = "Absorption line at 983 nm")
di = '../TRAITEMENT/LINES_bis/983_ABS'
make_dir(di)

plt.figure(figsize=(8,6))
a_983.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
a_983.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
a_983.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
a_983.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
a_983.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
a_983.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
a_983.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
a_983.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')

#plt.figure(figsize=(8,6))
#a_983.plot_pos(save_fig=True, save_nam=di+'/pos_0.3.png')

plt.figure(figsize=(8,6))
a_983.plot_prof_fit(step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[6,6,1,1])

plt.figure(figsize=(8,6))
a_983.plot_ew_fit(cont, step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[6,6,20,20])

plt.figure(figsize=(8,6))
a_983.plot_fit_gauss(-4,-3, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
a_983.plot_fit_gauss(3,4, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
a_983.plot_fit_gauss(-0.25,0.25, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
a_983.plot_fit_gauss(-1.5,-1, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
a_983.plot_fit_gauss(1,1.5, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

#%%
c_i=lines.make_line_obj(975,998, wvl=990.6, dble=983.82, name = "[C I] + [SVIII]")
di = '../TRAITEMENT/LINES_bis/986_EM'
make_dir(di)

plt.figure(figsize=(8,6))
c_i.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
c_i.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
c_i.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
c_i.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
c_i.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
c_i.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
c_i.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
c_i.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')

#plt.figure(figsize=(8,6))
#c_i.plot_pos(save_fig=True, save_nam=di+'/pos_0.3.png')

plt.figure(figsize=(8,6))
c_i.plot_prof_fit(step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[6,6,1,1])

plt.figure(figsize=(8,6))
c_i.plot_ew_fit(cont, step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[6,6,20,20])

plt.figure(figsize=(8,6))
c_i.plot_fit_gauss(-4,-3, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
c_i.plot_fit_gauss(3,4, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
c_i.plot_fit_gauss(-0.25,0.25, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
c_i.plot_fit_gauss(-1.5,-1, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
c_i.plot_fit_gauss(1,1.5, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')


#%%
he_ii=lines.make_line_obj(1001,1019, wvl=1012.35,  name = 'He II')
di = '../TRAITEMENT/LINES_bis/1012_HE_II'
make_dir(di)

plt.figure(figsize=(8,6))
he_ii.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
he_ii.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
he_ii.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
he_ii.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
he_ii.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
he_ii.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
he_ii.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
he_ii.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')

plt.figure(figsize=(8,6))
he_ii.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
he_ii.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
he_ii.plot_pos(step=0.05, save_fig=True,  speed=True, save_nam=di+'/pos_speed_0.05.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
he_ii.plot_pos(step=0.3, save_fig=True, speed=True, save_nam=di+'/pos_speed_0.3.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
he_ii.plot_prof_fit(step=0.3, title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[1.2,1.5,1,1])

plt.figure(figsize=(8,6))
he_ii.plot_ew_fit(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[1.2,1.5,20,20])

plt.figure(figsize=(8,6))
he_ii.plot_fit_gauss(-4,-3, title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
he_ii.plot_fit_gauss(3,4, title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
he_ii.plot_fit_gauss(-0.25,0.25, title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
he_ii.plot_fit_gauss(-1.5,-1, title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
he_ii.plot_fit_gauss(1,1.5, title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

#%%
s_ii=lines.make_line_obj(1020,1053, wvl=1033, name = '[S II]')
di = '../TRAITEMENT/LINES_bis/1032_S_II'
make_dir(di)

plt.figure(figsize=(8,6))
s_ii.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
s_ii.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
s_ii.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
s_ii.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
s_ii.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
s_ii.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
s_ii.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
s_ii.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')

plt.figure(figsize=(8,6))
s_ii.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
s_ii.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
s_ii.plot_pos(step=0.05, save_fig=True, speed=True, save_nam=di+'/pos_speed_0.05.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
s_ii.plot_pos(step=0.3, save_fig=True, speed=True, save_nam=di+'/pos_speed_0.3.png', params=[5,5,1,1])


plt.figure(figsize=(8,6))
s_ii.plot_prof_fit(step=0.3, title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
s_ii.plot_prof_fit(step=0.05, title='', save_fig=True, save_nam=di+'/prof_fit_0.05.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
s_ii.plot_ew_fit(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
s_ii.plot_fit_gauss(-4,-3, title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
s_ii.plot_fit_gauss(3,4, title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
s_ii.plot_fit_gauss(-0.25,0.25, title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
s_ii.plot_fit_gauss(-1.5,-1, title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
s_ii.plot_fit_gauss(1,1.5, title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')



#%%

#he_db=lines.make_line_obj(1070,1110, wvl=1083.03, dble=1092.4, name='He I')
#
#di = '../TRAITEMENT/LINES_bis/1083_HE_I'
#make_dir(di)
#
#plt.figure(figsize=(8,6))
#he_db.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_pos(step=0.05, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[2.2,2.2,3,3])
#
#plt.figure(figsize=(8,6))
#he_db.plot_pos(step=0.3, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[2.2,2.2,3,3])
#
#plt.figure(figsize=(8,6))
#he_db.plot_pos(step=0.05, mode='db_half_cons', speed=True, save_fig=True, save_nam=di+'/pos_speed_0.05.png', params=[2.6,2.2,3,3], title='Doppler shift of He I (1083 nm)')
#
#plt.figure(figsize=(8,6))
#he_db.plot_pos(step=0.3, mode='db_half_cons', speed=True, save_fig=True, save_nam=di+'/pos_speed_0.3.png', yaxis=2, params=[2.6,2.2,3,3], title='Doppler shift of He I (1083 nm)')
#
#plt.figure(figsize=(8,6))
#he_db.plot_prof_fit(step=0.3, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[6,6,1e-12,1e-12])
#
#plt.figure(figsize=(8,6))
#he_db.plot_prof_fit(step=0.05, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.05.png', params=[6,6,1e-12,1e-12])
#
#plt.figure(figsize=(8,6))
#he_db.plot_ew_fit(cont, step=0.3, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[6,6,20,20])
#
#plt.figure(figsize=(8,6))
#he_db.plot_fit_gauss(-4,-3, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_fit_gauss(3,4, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_fit_gauss(-0.25,0.25, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/gauss_center.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_fit_gauss(-1.5,-1, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')
#
#plt.figure(figsize=(8,6))
#he_db.plot_fit_gauss(1,1.5, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

#%%

pagam=lines.make_line_obj(1070,1110, wvl=1091, dble=1081, name='Paschen Gamma')

di = '../TRAITEMENT/LINES_bis/1093_PASCHEN_GAMMA'
make_dir(di)

plt.figure(figsize=(8,6))
pagam.plot_prof(1094, 1100, bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
pagam.plot_prof(1094, 1100, bin=0.3, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
pagam.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
pagam.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
pagam.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
pagam.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
pagam.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
pagam.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')
#
#plt.figure(figsize=(8,6))
#pagam.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[5,5,1,1])
#
#plt.figure(figsize=(8,6))
#pagam.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
pagam.plot_prof_fit(step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[6,6,1e-12,1e-12])

plt.figure(figsize=(8,6))
pagam.plot_prof_fit(step=0.05, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.05.png', params=[6,6,1e-12,1e-12])

plt.figure(figsize=(8,6))
pagam.plot_ew_fit(cont, step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[6,6,20,20])

plt.figure(figsize=(8,6))
pagam.plot_fit_gauss(-4,-3, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
pagam.plot_fit_gauss(3,4, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
pagam.plot_fit_gauss(-0.25,0.25, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
pagam.plot_fit_gauss(-1.5,-1, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
pagam.plot_fit_gauss(1,1.5, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

#%%
a_1111=lines.make_line_obj(1106,1128, wvl=1111, dble=1118, name = "Absorption line at 1111 nm")

di = '../TRAITEMENT/LINES_bis/1111_ABS'
make_dir(di)

plt.figure(figsize=(8,6))
a_1111.plot_prof(1094, 1113, bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
a_1111.plot_prof(1094, 1113, bin=0.3, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
a_1111.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
a_1111.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
a_1111.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
a_1111.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
a_1111.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
a_1111.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')
#
#plt.figure(figsize=(8,6))
#a_1111.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[5,5,1,1])
#
#plt.figure(figsize=(8,6))
#a_1111.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
a_1111.plot_prof_fit(step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[6,6,1e-12,1e-12])

plt.figure(figsize=(8,6))
a_1111.plot_prof_fit(step=0.05, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.05.png', params=[6,6,1e-12,1e-12])

plt.figure(figsize=(8,6))
a_1111.plot_ew_fit(cont, step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[6,6,20,20])

plt.figure(figsize=(8,6))
a_1111.plot_ew(cont, step=0.45, title='', save_fig=True, save_nam=di+'/ew_0.45.png')

plt.figure(figsize=(8,6))
a_1111.plot_ew(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_0.3.png')

plt.figure(figsize=(8,6))
a_1111.plot_ew(cont, step=0.05, title='', save_fig=True, save_nam=di+'/ew_0.05.png')

plt.figure(figsize=(8,6))
a_1111.plot_pos(step=0.05, speed=True, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/pos_speed_0.05.png', params=[5,5,2,2])

plt.figure(figsize=(8,6))
a_1111.plot_fit_gauss(-4,-3, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
a_1111.plot_fit_gauss(3,4, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
a_1111.plot_fit_gauss(-0.25,0.25, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
a_1111.plot_fit_gauss(-1.5,-1, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
a_1111.plot_fit_gauss(1,1.5, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

#%%
a_1118=lines.make_line_obj(1106,1128, wvl=1118, dble=1111, name = "Absorption line at 1118 nm")

di = '../TRAITEMENT/LINES_bis/1118_ABS'
make_dir(di)

plt.figure(figsize=(8,6))
a_1118.plot_prof(1113, 1125, bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
a_1118.plot_prof(1113, 1125, bin=0.3, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
a_1118.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
a_1118.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
a_1118.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
a_1118.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
a_1118.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
a_1118.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')

plt.figure(figsize=(8,6))
a_1118.plot_pos(step=0.05, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[5,5,2,2])

plt.figure(figsize=(8,6))
a_1118.plot_pos(step=0.3, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[5,5,2,2])

plt.figure(figsize=(8,6))
a_1118.plot_pos(step=0.05, speed=True, mode='db_half_cons', title='', save_fig=True, save_nam=di+'/pos_speed_0.05.png', params=[5,5,2,2])

plt.figure(figsize=(8,6))
a_1118.plot_prof_fit(step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[6,6,1e-12,1e-12])

plt.figure(figsize=(8,6))
a_1118.plot_prof_fit(step=0.05, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.05.png', params=[6,6,1e-12,1e-12])

plt.figure(figsize=(8,6))
a_1118.plot_ew_fit(cont, step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[6,6,20,20])

plt.figure(figsize=(8,6))
a_1118.plot_ew(cont, step=0.45, title='', save_fig=True, save_nam=di+'/ew_0.3=45.png')

plt.figure(figsize=(8,6))
a_1118.plot_ew(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_0.3.png')

plt.figure(figsize=(8,6))
a_1118.plot_ew(cont, step=0.05, title='', save_fig=True, save_nam=di+'/ew_0.05.png')

plt.figure(figsize=(8,6))
a_1118.plot_fit_gauss(-4,-3, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
a_1118.plot_fit_gauss(3,4, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
a_1118.plot_fit_gauss(-0.25,0.25, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
a_1118.plot_fit_gauss(-1.5,-1, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
a_1118.plot_fit_gauss(1,1.5, mode='db_cons', title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

#%%

uk_1135=lines.make_line_obj(1132,1145, wvl=1135, dble=1141)
a_1141=lines.make_line_obj(1132,1145, wvl=1141, dble=1135, name = "Absorption line at 1141 nm")
uk_1160=lines.make_line_obj(1154,1165, wvl=1160)


#%%

p_ii=lines.make_line_obj(1178,1198, wvl=1188.6, name='[P II]')
di = '../TRAITEMENT/LINES_bis/1188_P_II'
make_dir(di)

plt.figure(figsize=(8,6))
p_ii.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
p_ii.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
p_ii.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
p_ii.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
p_ii.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
p_ii.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
p_ii.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
p_ii.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')

plt.figure(figsize=(8,6))
p_ii.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[2,2,1,1])

plt.figure(figsize=(8,6))
p_ii.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[2,2,1,1])

plt.figure(figsize=(8,6))
p_ii.plot_pos(step=0.05, save_fig=True, speed=True, save_nam=di+'/pos_speed_0.05.png', params=[2,2,1,1])

plt.figure(figsize=(8,6))
p_ii.plot_pos(step=0.3, save_fig=True, speed=True, save_nam=di+'/pos_speed_0.3.png', params=[2,2,1,1])

plt.figure(figsize=(8,6))
p_ii.plot_prof_fit(step=0.3, title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[5,5,1e-13,1e-13])

plt.figure(figsize=(8,6))
p_ii.plot_prof_fit(step=0.05, title='', save_fig=True, save_nam=di+'/prof_fit_0.05.png', params=[5,5,1e-13,1e-13])

plt.figure(figsize=(8,6))
p_ii.plot_ew_fit(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[5,5,10,10])

plt.figure(figsize=(8,6))
p_ii.plot_fit_gauss(-4,-3, title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
p_ii.plot_fit_gauss(3,4, title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
p_ii.plot_fit_gauss(-0.25,0.25, title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
p_ii.plot_fit_gauss(-1.5,-1, title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
p_ii.plot_fit_gauss(1,1.5, title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

#%%

s_ix=lines.make_line_obj(1240,1273, wvl=1250, dble=1257, name="[S IX]")

di = '../TRAITEMENT/LINES_bis/1252_S_IX'
make_dir(di)

plt.figure(figsize=(8,6))
s_ix.plot_prof(1245, 1248, bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
s_ix.plot_prof(1245, 1248, bin=0.3, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
s_ix.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
s_ix.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
s_ix.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
s_ix.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
s_ix.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
s_ix.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')
#
#plt.figure(figsize=(8,6))
#s_ix.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[5,5,1,1])
#
#plt.figure(figsize=(8,6))
#s_ix.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
s_ix.plot_prof_fit(step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[3,3,1e-12,1e-12])

plt.figure(figsize=(8,6))
s_ix.plot_prof_fit(step=0.05, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.05.png', params=[3,3,1e-12,1e-12])

plt.figure(figsize=(8,6))
s_ix.plot_ew_fit(cont, step=0.3, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[6,6,20,20])

plt.figure(figsize=(8,6))
s_ix.plot_fit_gauss(-4,-3, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
s_ix.plot_fit_gauss(3,4, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
s_ix.plot_fit_gauss(-0.25,0.25, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
s_ix.plot_fit_gauss(-1.5,-1, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
s_ix.plot_fit_gauss(1,1.5, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')


#%%

fe_ii_1257=lines.make_line_obj(1240,1273, wvl=1257, dble=1250)

di = '../TRAITEMENT/LINES_bis/1257_FE_II'
make_dir(di)

plt.figure(figsize=(8,6))
fe_ii_1257.plot_prof(1257, 1260, bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_prof(1257, 1260, bin=0.3, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')
#
#plt.figure(figsize=(8,6))
#fe_ii_1257.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[5,5,1,1])
#
#plt.figure(figsize=(8,6))
#fe_ii_1257.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
fe_ii_1257.plot_prof_fit(step=0.3, mode='db_cons', params=[4,-1,1e-12,1e-12])
fe_ii_1257.plot_prof_fit(step=0.3, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[-.35,6,1e-12,1e-12])

plt.figure(figsize=(8,6))
#fe_ii_1257.plot_prof_fit(step=0.05, mode='db_cons', params=[4,-1,1e-12,1e-12])
#fe_ii_1257.plot_prof_fit(step=0.05, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.05.png', params=[-.35,6,1e-12,1e-12])
fe_ii_1257.plot_prof_fit(step=0.05, mode='db_cons', title='', save_fig=True, save_nam=di+'/prof_fit_0.05.png', params=[4,5,1e-12,1e-12])

plt.figure(figsize=(8,6))
fe_ii_1257.plot_ew_fit(cont, step=0.3, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[6,6,20,20])

plt.figure(figsize=(8,6))
fe_ii_1257.plot_fit_gauss(-4,-3, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_fit_gauss(3,4, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_fit_gauss(-0.25,0.25, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_fit_gauss(-1.5,-1, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
fe_ii_1257.plot_fit_gauss(1,1.5, mode='db_cons_pos', title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')


#%%
pabet=lines.make_line_obj(1263,1290, wvl=1282.1, name = 'Paschen Beta')
di = '../TRAITEMENT/LINES_bis/1282_PASCHEN_BETA'
make_dir(di)

plt.figure(figsize=(8,6))
pabet.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
pabet.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
pabet.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
pabet.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
pabet.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
pabet.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
pabet.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
pabet.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')

plt.figure(figsize=(8,6))
pabet.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[4,4,3,3])

plt.figure(figsize=(8,6))
pabet.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[4,4,3,3])

plt.figure(figsize=(8,6))
pabet.plot_pos(step=0.05, save_fig=True, speed=True, save_nam=di+'/pos_speed_0.05.png', params=[2,2,1,3])

plt.figure(figsize=(8,6))
pabet.plot_pos(step=0.3, save_fig=True, speed=True, save_nam=di+'/pos_speed_0.3.png', params=[2,2,1,3])

plt.figure(figsize=(8,6))
pabet.plot_prof_fit(step=0.3, title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[6,6,2e-13,1])

plt.figure(figsize=(8,6))
pabet.plot_ew_fit(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[6,6,10,20])

plt.figure(figsize=(8,6))
pabet.plot_fit_gauss(-4,-3, title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
pabet.plot_fit_gauss(3,4, title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
pabet.plot_fit_gauss(-0.25,0.25, title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
pabet.plot_fit_gauss(-1.5,-1, title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
pabet.plot_fit_gauss(1,1.5, title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

#%%
uk_1405=lines.make_line_obj(1400,1410, wvl=1405)


#%%
si_x=lines.make_line_obj(1423,1433, wvl=1430, name='[Si X]')
di = '../TRAITEMENT/LINES_bis/1430_SI_X'
make_dir(di)

plt.figure(figsize=(8,6))
si_x.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
si_x.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
si_x.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
si_x.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
si_x.plot_spec(3, 4, title='', save_fig=True, save_nam=di
               +'/spec_far_north.png')

plt.figure(figsize=(8,6))
si_x.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
si_x.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
si_x.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')

plt.figure(figsize=(8,6))
si_x.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png')

plt.figure(figsize=(8,6))
si_x.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png')

plt.figure(figsize=(8,6))
si_x.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
si_x.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
si_x.plot_prof_fit(step=0.3, title='', save_fig=True, save_nam=di+'/prof_fit_0.05.png', params=[2,2,2,2])

plt.figure(figsize=(8,6))
si_x.plot_ew_fit(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[1.2,1.5,20,20])

plt.figure(figsize=(8,6))
si_x.plot_fit_gauss(-4,-3, title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
si_x.plot_fit_gauss(3,4, title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
si_x.plot_fit_gauss(-0.25,0.25, title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
si_x.plot_fit_gauss(-1.5,-1, title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
si_x.plot_fit_gauss(1,1.5, title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

#%%
a_1463=lines.make_line_obj(1454,1473, wvl=1463, name = "Absorption line at 1463 nm")
di = '../TRAITEMENT/LINES_bis/1463_ABS'
make_dir(di)
plt.figure(figsize=(8,6))
a_1463.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
a_1463.plot_ew(cont, step=0.45, title='', save_fig=True, save_nam=di+'/ew_0.45.png')

plt.figure(figsize=(8,6))
a_1463.plot_pos(step=0.05, speed=True, save_fig=True, save_nam=di+'/pos_speed_0.05.png')

plt.figure(figsize=(8,6))
a_1463.plot_pos(step=0.05,  save_fig=True, save_nam=di+'/pos_0.05.png')

a_1463_b=copy(a_1463)
a_1463_b.obj = a_1463_b.obj-np.median(a_1463_b.obj[:,:20])

plt.figure(figsize=(8,6))
a_1463_b.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_b_0.05.png')

plt.figure(figsize=(8,6))
a_1463_b.plot_ew(cont, step=0.45, title='', save_fig=True, save_nam=di+'/ew_b_0.45.png')
#%%

a_1590=lines.make_line_obj(1575,1610, wvl = 1590, name = 'Absorption feature at 1590 nm')

di = '../TRAITEMENT/LINES_bis/1590_ABS'
make_dir(di)

plt.figure(figsize=(8,6))
a_1590.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
a_1590.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
a_1590.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
a_1590.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')
plt.figure(figsize=(8,6))
a_1590.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
a_1590.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
a_1590.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
a_1590.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')

plt.figure(figsize=(8,6))
a_1590.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png')
plt.figure(figsize=(8,6))
a_1590.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png')
#
#plt.figure(figsize=(8,6))
#a_1590.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[1.7,1.5,1,1])
#
#plt.figure(figsize=(8,6))
#a_1590.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
a_1590.plot_prof_fit(step=0.3, title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[1.2,1.5,1,1])

plt.figure(figsize=(8,6))
a_1590.plot_ew_fit(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[1.2,1.5,20,20])

plt.figure(figsize=(8,6))
a_1590.plot_fit_gauss(-4,-3, title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))/home/pvermot/Documents/2019/montagn_tycho_11_10_2019
a_1590.plot_fit_gauss(3,4, title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
a_1590.plot_fit_gauss(-0.25,0.25, title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
a_1590.plot_fit_gauss(-1.5,-1, title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
a_1590.plot_fit_gauss(1,1.5, title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

plt.figure(figsize=(8,6))
a_1590.plot_ew(cont, step=0.45, title='', save_fig=True, save_nam=di+'/ew_0.45.png')

plt.figure(figsize=(8,6))
a_1590.plot_ew(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_0.3.png')

plt.figure(figsize=(8,6))
a_1590.plot_ew(cont, step=0.05, title='', save_fig=True, save_nam=di+'/ew_0.05.png')


#%%
a_1620=lines.make_line_obj(1610,1630, wvl = 1620)
di = '../TRAITEMENT/LINES_bis/1620_ABS'
make_dir(di)

plt.figure(figsize=(8,6))
a_1620.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')
/home/pvermot/Documents/2019/montagn_tycho_11_10_2019
plt.figure(figsize=(8,6))
a_1620.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
a_1620.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
a_1620.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
a_1620.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
a_1620.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
a_1620.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
a_1620.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')
#
#plt.figure(figsize=(8,6))
#a_1620.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png')
#
#plt.figure(figsize=(8,6))
#a_1620.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png')

plt.figure(figsize=(8,6))
a_1620.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
a_1620.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
a_1620.plot_prof_fit(step=0.3, title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[1.2,1.5,1,1])

plt.figure(figsize=(8,6))
a_1620.plot_ew_fit(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[1.2,1.5,20,20])

plt.figure(figsize=(8,6))
a_1620.plot_fit_gauss(-4,-3, title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
a_1620.plot_fit_gauss(3,4, title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
a_1620.plot_fit_gauss(-0.25,0.25, title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
a_1620.plot_fit_gauss(-1.5,-1, title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
a_1620.plot_fit_gauss(1,1.5, title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

plt.figure(figsize=(8,6))
a_1620.plot_ew(cont, step=0.45, save_fig=True, save_nam=di+'/ew_0.45.png')

plt.figure(figsize=(8,6))
a_1620.plot_ew(cont, step=0.6, save_fig=True, save_nam=di+'/ew_0.6.png')

plt.figure(figsize=(8,6))
a_1620.plot_ew(cont, step=1, save_fig=True, save_nam=di+'/ew_1.0.png')

for k in range(1,20):
    plt.figure(k,figsize=(8,6))
    a_1620.plot_ew(cont, step=k/10)

plt.figure(figsize=(8,6))
a_1620.plot_ew(cont, step=0.3, save_fig=True, save_nam=di+'/ew_0.3.png')

plt.figure(figsize=(8,6))
a_1620.plot_ew(cont, step=0.05, save_fig=True, save_nam=di+'/ew_0.05.png')

plt.figure(figsize=(8,6))
a_1620.plot_ew(cont3, step=0.45, save_fig=True, save_nam=di+'/ew_0.45.png')
plt.title('')
#%%

fe_ii_1643=lines.make_line_obj(1630,1650, wvl= 1643)
di = '../TRAITEMENT/LINES_bis/1643_FE_II'
make_dir(di)

plt.figure(figsize=(8,6))
fe_ii_1643.plot_prof(bin=0.05, save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_prof(bin=0.1, save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_spec(-4, -3, save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_spec(3, 4, save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_spec(-0.25, 0.25, save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_spec(-1.5, -1, save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_spec(1, 1.5, save_fig=True, save_nam=di+'/spec_north.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_pos(step=0.05, save_fig=True, save_nam=di+'/pos_0.05.png', yaxis=3.5, params=[5,5.5,3,3])

plt.figure(figsize=(8,6))
fe_ii_1643.plot_pos(step=0.3, save_fig=True, save_nam=di+'/pos_0.3.png', yaxis=3.5, params=[5,5,3,3])

plt.figure(figsize=(8,6))
fe_ii_1643.plot_pos(step=0.05, save_fig=True, speed=True, save_nam=di+'/pos_speed_0.05.png', yaxis=5.5, params=[5,5.5,3,3], title='Doppler shift of Fe II (1643 nm)')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_pos(step=0.3, save_fig=True, speed=True,save_nam=di+'/pos_speed_0.3.png', yaxis=5.5, params=[5,5,3,3], title='Doppler shift of Fe II (1643 nm)')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_prof_fit(step=0.3, save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[5,5,1,1])

plt.figure(figsize=(8,6))
fe_ii_1643.plot_ew_fit(cont, step=0.3, save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[2,5,20,20])

plt.figure(figsize=(8,6))
fe_ii_1643.plot_fit_gauss(-4,-3, save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_fit_gauss(3,4, save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_fit_gauss(-0.25,0.25, title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_fit_gauss(-1.5,-1, title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
fe_ii_1643.plot_fit_gauss(1,1.5, title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')


#%%
a_1730=lines.make_line_obj(1720,1740, wvl = 1730)
di = '../TRAITEMENT/LINES_bis/1620_ABS'
make_dir(di)

plt.figure(figsize=(8,6))
a_1730.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
a_1730.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
a_1730.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
a_1730.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
a_1730.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
a_1730.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
a_1730.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
a_1730.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')
#
#plt.figure(figsize=(8,6))
#a_1730.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png')
#
#plt.figure(figsize=(8,6))
#a_1730.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png')

plt.figure(figsize=(8,6))
a_1730.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
a_1730.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
a_1730.plot_prof_fit(step=0.3, title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[1.2,1.5,1,1])

plt.figure(figsize=(8,6))
a_1730.plot_ew_fit(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[1.2,1.5,20,20])

plt.figure(figsize=(8,6))
a_1730.plot_fit_gauss(-4,-3, title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
a_1730.plot_fit_gauss(3,4, title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
a_1730.plot_fit_gauss(-0.25,0.25, title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
a_1730.plot_fit_gauss(-1.5,-1, title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
a_1730.plot_fit_gauss(1,1.5, title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

plt.figure(figsize=(8,6))
a_1730.plot_ew(cont, step=0.45, save_fig=True, save_nam=di+'/ew_0.45.png')

plt.figure(figsize=(8,6))
a_1730.plot_ew(cont, step=0.6, save_fig=True, save_nam=di+'/ew_0.6.png')

plt.figure(figsize=(8,6))
a_1730.plot_ew(cont, step=1, save_fig=True, save_nam=di+'/ew_1.0.png')

for k in range(1,20):
    plt.figure(k,figsize=(8,6))
    a_1730.plot_ew(cont, step=k/10)

plt.figure(figsize=(8,6))
a_1730.plot_ew(cont, step=0.3, save_fig=True, save_nam=di+'/ew_0.3.png')

plt.figure(figsize=(8,6))
a_1730.plot_ew(cont, step=0.05, save_fig=True, save_nam=di+'/ew_0.05.png')

plt.figure(figsize=(8,6))
a_1730.plot_ew(cont3, step=0.45, save_fig=True, save_nam=di+'/ew_0.45.png')
plt.title('')
#%%
#%%
a_1710=lines.make_line_obj(1700,1720, wvl = 1710)
di = '../TRAITEMENT/LINES_bis/1620_ABS'
make_dir(di)

plt.figure(figsize=(8,6))
a_1710.plot_prof(bin=0.05, title='', save_fig=True, save_nam=di+'/prof_0.05.png')

plt.figure(figsize=(8,6))
a_1710.plot_prof(bin=0.1, title='', save_fig=True, save_nam=di+'/prof_0.3.png')

plt.figure(figsize=(8,6))
a_1710.plot_spec(save_fig=True, save_nam=di+'/spec_full.png')

plt.figure(figsize=(8,6))
a_1710.plot_spec(-4, -3, title='', save_fig=True, save_nam=di+'/spec_far_south.png')

plt.figure(figsize=(8,6))
a_1710.plot_spec(3, 4, title='', save_fig=True, save_nam=di+'/spec_far_north.png')

plt.figure(figsize=(8,6))
a_1710.plot_spec(-0.25, 0.25, title='', save_fig=True, save_nam=di+'/spec_center.png')

plt.figure(figsize=(8,6))
a_1710.plot_spec(-1.5, -1, title='', save_fig=True, save_nam=di+'/spec_south.png')

plt.figure(figsize=(8,6))
a_1710.plot_spec(1, 1.5, title='', save_fig=True, save_nam=di+'/spec_north.png')
#
#plt.figure(figsize=(8,6))
#a_1710.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png')
#
#plt.figure(figsize=(8,6))
#a_1710.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png')

plt.figure(figsize=(8,6))
a_1710.plot_pos(step=0.05, title='', save_fig=True, save_nam=di+'/pos_0.05.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
a_1710.plot_pos(step=0.3, title='', save_fig=True, save_nam=di+'/pos_0.3.png', params=[1.7,1.5,1,1])

plt.figure(figsize=(8,6))
a_1710.plot_prof_fit(step=0.3, title='', save_fig=True, save_nam=di+'/prof_fit_0.3.png', params=[1.2,1.5,1,1])

plt.figure(figsize=(8,6))
a_1710.plot_ew_fit(cont, step=0.3, title='', save_fig=True, save_nam=di+'/ew_fit_0.3.png', params=[1.2,1.5,20,20])

plt.figure(figsize=(8,6))
a_1710.plot_fit_gauss(-4,-3, title='', save_fig=True, save_nam=di+'/gauss_fit_far_south.png')

plt.figure(figsize=(8,6))
a_1710.plot_fit_gauss(3,4, title='', save_fig=True, save_nam=di+'/gauss_fit_far_north.png')

plt.figure(figsize=(8,6))
a_1710.plot_fit_gauss(-0.25,0.25, title='', save_fig=True, save_nam=di+'/gauss_center.png')

plt.figure(figsize=(8,6))
a_1710.plot_fit_gauss(-1.5,-1, title='', save_fig=True, save_nam=di+'/gauss_fit_south.png')

plt.figure(figsize=(8,6))
a_1710.plot_fit_gauss(1,1.5, title='', save_fig=True, save_nam=di+'/gauss_fit_north.png')

plt.figure(figsize=(8,6))
a_1710.plot_ew(cont, step=0.45, save_fig=True, save_nam=di+'/ew_0.45.png')

plt.figure(figsize=(8,6))
a_1710.plot_ew(cont, step=0.6, save_fig=True, save_nam=di+'/ew_0.6.png')

plt.figure(figsize=(8,6))
a_1710.plot_ew(cont, step=1, save_fig=True, save_nam=di+'/ew_1.0.png')

for k in range(1,20):
    plt.figure(k,figsize=(8,6))
    a_1710.plot_ew(cont, step=k/10)

plt.figure(figsize=(8,6))
a_1710.plot_ew(cont, step=0.3, save_fig=True, save_nam=di+'/ew_0.3.png')

plt.figure(figsize=(8,6))
a_1710.plot_ew(cont, step=0.05, save_fig=True, save_nam=di+'/ew_0.05.png')

plt.figure(figsize=(8,6))
a_1710.plot_ew(cont3, step=0.45, save_fig=True, save_nam=di+'/ew_0.45.png')
plt.title('')
#%%
"""
"""
#%%
#
plt.clf()
fe_16 = fe_ii_1643.make_prof(bin=0.05)
fe_12 = fe_ii_1257.plot_prof_fit(step=0.05, mode='db_cons', params=[4,5,1e-12,1e-12])
p_2 = p_ii.make_prof(bin=0.05)

fe_16[1] -= fe_16[1][-1]
p_2[1] -= p_2[1][-1]


pos = fe_16[0]
where = np.where(((np.array(pos)>-3)*(np.array(pos)<3))==1)
ratio =  fe_16[1]/p_2[1]
er = ((fe_16[2]/p_2[1])**2+(fe_16[1]*p_2[2]/p_2[1]**2)**2)**0.5
plt.plot(pos[where], ratio[where])
plt.fill_between(pos[where], ratio[where]-er[where], ratio[where]+er[where], color = 'b', alpha=0.3)
plt.xlabel('Position relative to photocenter (arcsecond)')
plt.ylabel(r'$\left[Fe\ II\right]\ \lambda 1643\ /\ \left[P\ II\right] $')
plt.plot(fe_12[0][fe_12[-1][0]],8.14*np.log10(1.36/((fe_12[1][fe_12[-1][0]]/(fe_16[1][fe_12[-1][0]]+0.125e-14)))))

"""
"""
cont3.reset_mask()
result = []
ages = np.arange(20,200,4)
temps = np.arange(700,910,10)
for age in ages:
    for temp in temps:
        print(age, temp)
        try:
            result.append([age, temp, cont3.plot_fluxs_res(cont3.obj[:,cont3.ptx(-2):cont3.ptx(2)], ['SSP-'+str(int(age)), 'BB'+str(int(temp)), 'PL0'],  mean=25)])
        except:
            print('Error')
            result.append(result[-1])

#%%
min_temp = 1
resu_temp = []
for resu in result:
    if resu[2]<min_temp:
        min_temp=resu[2]
        resu_temp = resu

print(min_temp, resu_temp)

#%%

result = np.array(result)
tab = np.zeros((len(ages),len(temps)))

i, j = 0, 0
age_temp = result[0][0]
for resu in result:
    if resu[0] == age_temp:
        tab[i,j] = resu[2]
        j += 1
        age_temp = resu[0]
    else:
        i += 1
        j = 0
        tab[i,j] = resu[2]
        j += 1
        age_temp = resu[0]

from matplotlib.colors import LogNorm
plt.imshow(tab, extent = [np.min(result[:,1]), np.max(result[:,1]), np.max(result[:,0]), np.min(result[:,0])], norm=LogNorm(), aspect='auto', vmax=1e-12)
#plt.clim(4e-12,5e-12)
plt.colorbar()
plt.xlabel('Temperature (K)')
plt.ylabel('Age of stellar population (Myr)')
#%%

#min_temp = 1
#resu_temp = []
#for resu in result:
#    print(resu)
#    if resu[2]<min_temp:
#        min_temp=resu[2]
#        resu_temp = resu
#
#print(min_temp, resu_temp)
#
##%%
#
#tab = np.zeros((21, 33))
#
#i, j = 0, 0
#age_temp = result[0][0]
#for resu in result:
#    print(i,j)
#    print(resu)
#    if resu[0] is age_temp:
#        tab[i,j] = resu[2]
#        j += 1
#        age_temp = resu[0]
#    else:
#        i += 1
#        j = 0
#        tab[i,j] = resu[2]
#        j += 1
#        age_temp = resu[0]
#
#from matplotlib.colors import LogNorm
#plt.imshow(tab, extent = [400, 1200, 160, 80], norm=LogNorm(), aspect='auto')
#plt.clim(4e-12,6e-12)
#plt.colorbar()
#plt.xlabel('Temperature (K)')
#plt.ylabel('Age of stellar population (Myr)')
"""
#%%
"""
ress=[]
for age in np.arange(4,1000,4):
    print('SSP-'+str(age))
    print(cont2.fit_props(np.mean(cont2.obj[:,cont2.ptx(1):cont2.ptx(2)],1),['SSP-'+str(age)]))
    ress.append(cont3.fit_props(np.mean(cont3.obj[:,cont2.ptx(1.5):cont3.ptx(2.5)],1),['SSP-'+str(age)])[1])

plt.plot(np.arange(4,1000,4), ress)
plt.xlabel("Age of the stellar population (Myr)")
plt.ylabel("Residuals (Arbitrary unit)")
#%%
cont3.load_bbs(T_min=100, T_max=1000, step=1)
ress=[]
for temp in np.arange(400,1000, 1):
#    print('SSP-'+str(age))
#    print(cont2.fit_props(np.mean(cont2.obj[:,cont2.ptx(1):cont2.ptx(2)],1),['SSP-'+str(age)]))
    ress.append(cont3.fit_props(np.mean(cont3.obj[:,cont2.ptx(-0.05):cont3.ptx(.15)],1),['SSP-120', 'BB'+str(temp)])[1])

plt.plot(np.arange(400,1000, 1), ress)
plt.xlabel("Temperature of the blackbody component (K)")
plt.ylabel("Residuals (Arbitrary unit)")
"""
#%%

"""
#cont3.load_pls(e_min=100, T_max=1000, step=1)
ress=[]
for pl in np.arange(-400,100, 1):
#    print('SSP-'+str(age))
#    print(cont2.fit_props(np.mean(cont2.obj[:,cont2.ptx(1):cont2.ptx(2)],1),['SSP-'+str(age)]))
    print(pl)
    ress.append(cont3.fit_props(np.mean(cont3.obj[:,cont2.ptx(-3):cont3.ptx(-2)],1),['SSP-120', 'BB830', 'PL'+str(pl)])[1])

plt.plot(np.arange(-400,100, 1), ress)
plt.xlabel("100 x power of the powerlaw")
plt.ylabel("Residuals (Arbitrary unit)")
#%%

cont = ngc.plot_raw_cont_1D_int(-5, 5, lambdas=ngc.lam[np.logical_not(ngc.obj.mask[:,683])],deg=3, fig=33, std_data=False)
em_lines_names = ['[S VIII]', 'He II', '[S II]', r'He I + Pa $\gamma$', '[P II]', '[S IX]', r'[Fe II] + Pa $\beta$', '[Si X]', '[Fe II]']
abs_lines_names = ['1111 + 1118', '1590', '1620']

#em_lines_lam = [991,1012,1032,1085,1188,1255,1280,1430,1643]
abs_lines_lam = [1114.5,1590,1620]

def plot_text(liname, linlam, spec, color='k'):
    maxi = np.max(spec)
    print(maxi)
    for i in range(len(liname)):
        plt.scatter(linlam[i], 2.6*maxi, color='white')
        plt.text(linlam[i], 2.1*maxi, liname[i], rotation = 90, fontsize=9, ha='center', va='bottom', color=color)
        plt.plot([linlam[i], linlam[i]],[1.2*cont[ngc.ltx(linlam[i])], 2.0*maxi], linestyle=':', color='k', alpha=0.5)

plot_text(em_lines_names, em_lines_lam, cont, color='green')
plot_text(abs_lines_names, abs_lines_lam, cont, color='red')
plt.legend(loc="right")
plt.title('')
"""

#%%


"""
import csv

def read_csv(file):
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            line_count += 1
    return 

import pandas
df = pandas.read_csv('/home/pvermot/Documents/2021/CLOUDY/nist.csv', delimiter=',')
print(df)
ls_obs = np.array([1707.647,1730.766])
ls_nist = np.array(df['obs_wl_air(nm)'])

est_min = 10
for v in np.arange(0,-2000,-1):
    z = v*1e3/3e8
    ls = ls_obs*(1+z)
    a = np.min((ls_nist-ls[0])**2/df['fik'])
    i = np.argmin((ls_nist-ls[0])**2/df['fik'])
    b = np.min((ls_nist-ls[1])**2/df['fik'])
    j = np.argmin((ls_nist-ls[1])**2/df['fik'])
    est = (a+b)**0.5
    if est < est_min:
        est_min = est
        print(df['element'][i],df['sp_num'][i], ls[0], ls_nist[i] ,df['element'][j],df['sp_num'][j], ls[1], ls_nist[j], est, z, v)

#%%


ls_obs = np.array([1707.647,1730.766, 1681.473, 1657.388])

dft = pandas.read_csv('/home/pvermot/Documents/2021/CLOUDY/nist_long.csv', delimiter=',')
df = dft[dft['fik']>1e-9]
df = df.reset_index()
ls_nist = np.array(df['obs_wl_air(nm)'])
est_min = 1e6
with_fik = False

for v in np.arange(0,-200,-1):
    z = v*1e3/3e8
    ls = ls_obs*(1+z)
    inds = []
    els = []
    ls_ni = []
    est = 0
    for l in ls:
        if with_fik:
            est += np.min((ls_nist-l)**2/df['fik']**2)
            i = np.argmin((ls_nist-l)**2/df['fik']**2)
        else:
            est += np.min((ls_nist-l)**2)
            i = np.argmin((ls_nist-l)**2)
        inds.append(i)
        ls_ni.append(df['obs_wl_air(nm)'][i])
        els.append(df['element'][i])
        est = est**0.5
    if est < est_min:
        est_min = est
        print(' ')
        print('#########################')
        print(' ')
        print('Speed : ', v)
        print('Estimator : ', est)
        print('redshift : ', z)
        print('Lines : ', els)
        print('Observed wavelengths : ', ls)
        print('NIST catalog wavelen : ', ls_ni)
        
#%%

def correlation_lags(in1_len, in2_len, mode='full'):

    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags

#%%

im = ngc1068.obj.data
smoothed_im = scipy.signal.medfilt(im, kernel_size=3)
dif = abs(im-smoothed_im)
new_mask = (dif+1e-20)>3e-19
ngc1068.obj.mask += new_mask
ngc1068.obj = interpolate_masked_array(ngc1068.obj)
ngc = deepcopy(ngc1068)

l0, s0, st0 = ngc.make_spec(0.15,.25)
ln, sn, stn = ngc.make_spec(0.25,.34)
l1, s1, st1 = ngc.make_spec(0.34,.44)
snod = (2*sn-s0-s1)/2
std_nod = ((2*stn)**2+st0**2+st1**2)**0.5/2

l0, cs0, cst0 = cont.make_spec(0.15,.25)
ln, csn, cstn = cont.make_spec(0.25,.34)
l1, cs1, cst1 = cont.make_spec(0.34,.44)
cont_snod = (2*csn-cs0-cs1)/2
cont_std_nod = ((2*cstn)**2+cst0**2+cst1**2)**0.5/2

l2, s2, st2 = ngc.make_spec(0.6,.7)
ln2, sn2, stn2 = ngc.make_spec(0.7,.8)
l3, s3, st3 = ngc.make_spec(0.8,.9)
snod2 = (2*sn2-s2-s3)/2
std_nod2 = ((2*stn2)**2+st2**2+st3**2)**0.5/2

l2, cs2, cst2 = cont.make_spec(0.6,.7)
ln2, csn2, cstn2 = cont.make_spec(0.7,.8)
l3, cs3, cst3 = cont.make_spec(0.8,.9)
cont_snod2 = (2*csn2-cs2-cs3)/2
cont_std_nod2 = ((2*cstn2)**2+cst2**2+cst3**2)**0.5/2

plt.figure()
plt.plot(l1, snod-cont_snod, label='r=0.3" nodule')
plt.plot(l1, (snod2-cont_snod2)*6.25*0.57, label='r=0.75" nodule')
plt.legend()

plt.figure()
plt.plot(l1, snod, c='k', label='r=0.3" nodule')
plt.plot(l1, cont_snod, c='k')
plt.legend()
plt.figure()
plt.plot(l2, snod2, c='r', label='r=0.75" nodule')
plt.plot(l2, cont_snod2, c='r')
plt.legend()
# plt.plot(l1, snod2)
#%%
from scipy import signal
import scipy

corr = signal.correlate(snod,snod2)
lag = correlation_lags(len(snod), len(snod2))


def gauss(x, x_0, amp, alpha, a, b):
    return amp*np.exp(-alpha*(x-x_0)**2)+a*x+b

p1, c1 = scipy.optimize.curve_fit(gauss, l1, snod-cont_snod, p0 =[1080, np.max(snod), 1, 0,0])
p2, c2 = scipy.optimize.curve_fit(gauss, l1, (snod2-cont_snod2)*6, p0 =[1080, np.max(snod), 1, 0,0])


v = (p1[0]-p2[0])*3e8/(np.mean([p1[0], p2[0]]))

print(v)

#%%

def gauss_1D(lam, lam0, fwhm):
    c = fwhm/2.35
    return np.exp(-(lam-lam0)**2/(2*c**2))

def em_line(lam, v):
    c = 3e8
    lam0 = 1100
    lam1 = lam0*(1+v/c)
    x = np.argmin((lam-lam1)**2)
    spec = np.zeros(len(lam))
    spec[x]=1
    return spec

x = np.arange(-600,600)
y = np.arange(-600,600)

xx, yy = np.meshgrid(x,y)

r = (xx**2+yy**2)**0.5

shell = (r>490) * (r<=500)


vs = [50, 100, 300, 500, 1500]
fig, axes = plt.subplots(5, 5, figsize=(15, 15))
i0=0
for v0 in vs:
    v = v0*1e3*shell*yy/500
    
    lam = np.arange(1090,1110,0.01)
    spec = np.zeros(len(lam))
    for i in range(len(x)):
        for j in range(len(y)):
            v0 = v[i,j]
            if v0 != 0:
                spec += em_line(lam, v[i,j])
            
    # plt.imshow(v)
    # plt.figure()
    j0 = 0
    for r in [350, 1000, 3000, 5000, 10000]:
        ax = axes[i0, j0]
        j0 += 1
        ax.set_title("V = "+str(int(np.max(v)/1e3))+'; R ='+str(int(r)) )
        ax.plot(lam, np.convolve(spec, gauss_1D(lam, 1100, 1100/r), mode='same'))
    i0 += 1
    plt.tight_layout()

"""