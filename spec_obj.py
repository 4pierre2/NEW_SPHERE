#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 16:20:41 2018

@author: pvermot
"""

import glob
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy.signal
import sys
from copy import copy
sys.path.append('../REDUCTION/CODE/')

from matplotlib.colors import LogNorm
#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
from cycler import cycler
#plt.rcParams['axes.prop_cycle'] = cycler(color='kbrgymgmc')
from scipy.optimize import curve_fit



def gauss(x, x_0, amp, alpha, a, b):
    return amp*np.exp(-alpha*(x-x_0)**2)+a*x+b

def alpha_to_fwhm(alpha):
    return 2*(2*np.log(2))**0.5*(1/(2*alpha))**0.5

def db_gauss(x, x_0, x_1, amp_0, amp_1, alpha, a, b):
    return amp_0*np.exp(-alpha*(x-x_0)**2)+amp_1*np.exp(-alpha*(x-x_1)**2)+a*x+b

def rebin(x, bin, mode='mean'):
    r = len(x)%bin
    wei=[]
    for j in range(bin):
        if r == 0:
            wei.append(x[j::bin])
        else:
            wei.append(x[j:-r:bin])
    if mode == 'mean':
        x_bin=np.mean(np.array(wei),0)
    if mode == 'sum':
        x_bin=np.sum(np.array(wei),0)
    if mode == 'std_sum':
        x_bin=np.sum(np.array(wei)**2,0)**0.5
    if mode == 'std_mean':
        x_bin=np.sum(np.array(wei)**2,0)**0.5/bin
    return x_bin

def rebin(data, bins, mode='sum'):
    if mode == 'mean':
        result = data[:(data.size // bins) * bins].reshape(-1, bins).mean(axis=1)
    if mode == 'sum':
        result = data[:(data.size // bins) * bins].reshape(-1, bins).sum(axis=1)
    return result

def bbrad(lam,temp):
    h=6.626070040*10**-34
    kb=1.38064852*10**-23
    c=299792458
    B=((2*h*c**2)/(lam**5))*(1/(np.exp((h*c)/(lam*kb*temp))-1))
    return B*10**-10

def bbrad_p(lam,temp,a):
    h=6.626070040*10**-34
    kb=1.38064852*10**-23
    c=299792458
    B=a*((2*h*c**2)/(lam**5))*(1/(np.exp((h*c)/(lam*kb*temp))-1))
    return B*10**-10

def interpolate_masked_array(array, method='linear'):
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

class Spec_Obj():
    def __init__(self, obj_final, std_obj_final, lam, pos):
        self.obj_nm = obj_final
        self.std_nm = std_obj_final
        self.mask = np.zeros(np.shape(self.obj_nm))
        self.obj = ma.masked_array(obj_final, self.mask)
        self.std = ma.masked_array(std_obj_final, self.mask)
        self.lam = lam/1.0038#25
        self.pos = pos
        self.pix_scale = np.abs(np.median((np.roll(self.pos,1)-self.pos)[2:-2]))
        self.pos = pos
        self.lam_pos = np.meshgrid(self.lam, self.pos)
        self.x = np.arange(len(self.obj_nm[:,0]))
        self.y = np.arange(len(self.obj_nm[0,:]))
        self.xy = np.meshgrid(self.x, self.y)
        self.pix_scale = np.abs(np.mean((np.roll(self.pos,1)-self.pos)[2:-2]))
        self.pix_bw = np.abs(np.mean((np.roll(self.lam,1)-self.lam)[2:-2]))



    def update_mask(self, new_mask):
        self.mask = new_mask
        self.obj = ma.masked_array(self.obj.data, self.mask)
        self.std = ma.masked_array(self.std.data, self.mask)

    def reset_mask(self):
        new_mask = copy(self.mask)
        new_mask *= 0
        self.mask = new_mask
        self.obj = ma.masked_array(self.obj.data, self.mask)
        self.std = ma.masked_array(self.std.data, self.mask)

    def ptx(self, posi):
        return np.argmin((self.pos-posi)**2)
    def ltx(self, lamb):
        return np.argmin((self.lam-lamb)**2)


#%%


    def imshow(self, title='2D spectrum', save_fig=False, save_nam='./', lognorm=False):
        extent=[self.pos[0], self.pos[-1], self.lam[0], self.lam[-1]]
        aspect=len(self.pos)*abs(extent[3]-extent[2])/(abs(extent[1]-extent[0])*len(self.lam))
        if lognorm:
            plt.imshow((self.obj/self.pix_bw/self.pix_scale/1e-3), extent=extent, interpolation=None, aspect=1/aspect, norm=LogNorm(), origin='lower')
        else:
            plt.imshow((self.obj/self.pix_bw/self.pix_scale/1e-3), extent=extent, interpolation=None, aspect=1/aspect, origin='lower')
        cbar = plt.colorbar()
        cbar.set_label(r'Flux ($W.m^{-2}.\mu m^{-1}.arcsec^{-1}$)',size=11)
        plt.xlabel('Position relative to photcenter (arcsecond)', fontsize=11)
        plt.ylabel('Wavelength (nm)', fontsize=11)
        plt.title(title, fontsize=11, fontweight="bold")
        if save_fig:
            plt.savefig(save_nam)
            plt.close()

    def make_prof(self, lam_0='default', lam_1='default', bins=False, centered=True):
        if lam_0 == 'default':
            lam_0 = np.min(self.lam)
        if lam_1 == 'default':
            lam_1 = np.max(self.lam)
        pos = self.pos[:]
        prof=np.sum(self.obj[self.ltx(lam_0):self.ltx(lam_1)+1],0)/self.pix_scale
        std=np.sum((self.std[self.ltx(lam_0):self.ltx(lam_1)+1]/self.pix_scale)**2,0)**0.5
        
        if bins:
            pos = rebin(pos, bins, mode='mean')
            prof = rebin(prof, bins, mode='mean')#prof[:(prof.size // bins) * bins].reshape(-1, bins).sum(axis=1)
            std = rebin(std**2, bins, mode='sum')**0.5/bins#((std[:(std.size // bins) * bins].reshape(-1, bins)**2).sum(axis=1))**0.5

        return pos, prof, std

    def make_spec(self, pos_0='default', pos_1='default', bins=False):
        if pos_0 == 'default':
            pos_0 = np.min(self.pos)
        if pos_1 == 'default':
            pos_1 = np.max(self.pos)
            
        lams = self.lam[:]
        spec = np.sum(self.obj[:,self.ptx(pos_0):self.ptx(pos_1)+1],1)/self.pix_bw/1e-3
        std = np.sum((self.std[:,self.ptx(pos_0):self.ptx(pos_1)+1]/self.pix_bw/1e-3)**2,1)**0.5
        
        if bins:
            lams = rebin(lams, bins, mode='mean')
            spec = rebin(spec, bins, mode='mean')#prof[:(prof.size // bins) * bins].reshape(-1, bins).sum(axis=1)
            std = rebin(std**2, bins, mode='sum')**0.5/bins#((std[:(std.size // bins) * bins].reshape(-1, bins)**2).sum(axis=1))**0.5


        return lams, spec, std
    
    def make_interp_spec(self, p0a, p0b, p1a, p1b, p2a, p2b):
        l, s0, e0 = self.make_spec(p0a, p0b)
        l, s1, e1 = self.make_spec(p1a, p1b)
        l, s2, e2 = self.make_spec(p2a, p2b)
        x0a = self.ptx(p0a)
        x0b = self.ptx(p0b)
        x1a = self.ptx(p2a)
        x1b = self.ptx(p2b)
        x2a = self.ptx(p2a)
        x2b = self.ptx(p2b)
        s = s1-(x1b-x1a)*(s0/(x0b-x0a)+s2/(x2b-x2a))/2
        e = e1+(x1b-x1a)*(e0/(x0b-x0a)+e2/(x2b-x2a))/2
        return l, s, e
    
    def make_tight_spec_interp(self, p0, p1, n_pix=2):
        x1a = self.ptx(p0)
        x1b = self.ptx(p1)
        x0a = x1a-n_pix
        x0b = x1a-1
        x2a = x1b+1
        x2b = x1b+n_pix
        p0a = self.pos[x0a]
        p0b = self.pos[x0b]
        p1a = self.pos[x1a]
        p1b = self.pos[x1b]
        p2a = self.pos[x2a]
        p2b = self.pos[x2b]
        l, s, e = self.make_interp_spec(p0a, p0b, p1a, p1b, p2a, p2b)
        return l, s, e
        
        
    def plot_prof(self, lam_0='default', lam_1='default', title='default', save_fig=False, save_nam='./', lognorm=False, bins=False, color='k', color_err='red', legend=''):
        if lam_0 == 'default':
            lam_0 = np.min(self.lam)
        if lam_1 == 'default':
            lam_1 = np.max(self.lam)
        pos, prof, std = self.make_prof(lam_0, lam_1, bins)
        plt.fill_between(pos, prof+std, prof-std, color=color_err, alpha=0.3, step='mid')
        if legend is '':
            plt.step(pos, prof, color=color, where='mid')
        else:
            plt.step(pos, prof, color=color, where='mid',label=legend)
        if lognorm == 'x':
            plt.semilogx()
        if lognorm == 'y':
            plt.semilogy()
        if lognorm == 'xy':
            plt.loglog()
        plt.xlabel('Distance to photocenter (arcsecond)', fontsize=11)
        plt.ylabel('Flux ($W.m^{-2}.arcsecond^{-1}$)', fontsize=11)
        if title == 'default':
            title = 'Spatial profile integrated from '+str(int(lam_0))+' nm to '+str(int(lam_1))+' nm'
        plt.title(title, fontsize=11, fontweight="bold")
        if not legend is '':
            plt.legend()
        if save_fig:
            plt.savefig(save_nam)
            plt.close()
        return prof
    
    def plot_spec(self, pos_0='default', pos_1='default', title='default', save_fig=False, save_nam='./', lognorm=False, bins=False, step=False, color_err='red'):
        if pos_0 == 'default':
            pos_0 = np.min(self.pos)
        if pos_1 == 'default':
            pos_1 = np.max(self.pos)
        lam, spec, std = self.make_spec(pos_0, pos_1, bins=bins)
        if step:
            plt.fill_between(lam, spec+std, spec-std, color=color_err, alpha=0.3, step='mid')
            plt.step(lam, spec, color='k', where='mid')
        else:
            plt.fill_between(lam, spec+std, spec-std, color=color_err, alpha=0.3)
            plt.plot(lam, spec, color='k')
            
        if lognorm == 'x':
            plt.semilogx()
        if lognorm == 'y':
            plt.semilogy()
        if lognorm == 'xy':
            plt.loglog()
        plt.xlabel('Wavelength (nm)', fontsize=11)
        plt.ylabel('Flux ($W.m^{-2}.\mu m^{-1}$)', fontsize=11)
        if title == 'default':
            title = 'Spectrum integrated from '+str(int(100*pos_0)/100)+'" to '+str(int(100*pos_1)/100)+'"'
        plt.title(title, fontsize=11, fontweight="bold")
        if save_fig:
            plt.savefig(save_nam)
            plt.close()
        return spec

    def get_mag(self, pos_0='default', pos_1='default', band='H'):
        if pos_0 == 'default':
            pos_0 = np.min(self.pos)
        if pos_1 == 'default':
            pos_1 = np.max(self.pos)
        x0 = self.ptx(pos_0)
        x1 = self.ptx(pos_1)
        
        h_filt = np.genfromtxt("./ADDITIONAL_DATA/H_FILTER_2MASS.dat", delimiter='  ')
        h_f = np.interp(self.lam, h_filt[:, 0]*10**3, h_filt[:, 1])
        j_filt = np.genfromtxt("./ADDITIONAL_DATA/J_FILTER_2MASS.dat", delimiter='  ')
        j_f = np.interp(self.lam, j_filt[:, 0]*10**3, j_filt[:, 1])

        spec = np.sum(self.obj[:, x0:x1+1], 1)
        err = np.var(self.obj[:, x0:x1+1], 1)*(x1-x0)
        if band == 'H':
            flux = np.sum(spec*h_f)
            flux_per_mic_cm2 = flux/0.251/1e4
            err = np.sum(err*h_f)**0.5
            err_per_mic_cm2 = err/0.251/1e4
            mag = -2.5*np.log10(flux_per_mic_cm2/1.133e-13)
            mag_err_p = -2.5*np.log10((flux_per_mic_cm2+err_per_mic_cm2)/1.133e-13)
            mag_err_m = -2.5*np.log10((flux_per_mic_cm2-err_per_mic_cm2)/1.133e-13)
        elif band== 'J':
            flux = np.sum(spec*j_f)
            flux_per_mic_cm2 = flux/0.162/1e4
            err = np.sum(err*j_f)**0.5
            err_per_mic_cm2 = err/0.162/1e4
            mag = -2.5*np.log10(flux_per_mic_cm2/3.129e-13)
            mag_err_p = -2.5*np.log10((flux_per_mic_cm2+err_per_mic_cm2)/3.129e-13)
            mag_err_m = -2.5*np.log10((flux_per_mic_cm2-err_per_mic_cm2)/3.129e-13)
            
        return mag_err_p, mag, mag_err_m
    

    def plot_prof_spec(self, lam_0='default', lam_1='default', pos_0='default', pos_1='default',  title_prof='default', title_spec='default', save_fig=False, save_nam='./', lognorm_prof=False,  lognorm_spec=False, bin_prof=False, bin_spec=False):
        if lam_0 == 'default':
            lam_0 = np.min(self.lam)
        if lam_1 == 'default':
            lam_1 = np.max(self.lam)
        if pos_0 == 'default':
            pos_0 = np.min(self.pos)
        if pos_1 == 'default':
            pos_1 = np.max(self.pos)
        plt.figure(figsize=(21,5))
        plt.subplot(121)
        self.plot_prof(lam_0=lam_0, lam_1=lam_1, title=title_prof, lognorm=lognorm_prof, bin=bin_prof)
        plt.subplot(122)
        self.plot_spec(pos_0=pos_0, pos_1=pos_1, title=title_spec, lognorm=lognorm_spec, bin=bin_spec)
        s = self.make_spec(pos_0=pos_0, pos_1=pos_1, bin=bin_spec)
        y_min, y_max = np.min(s[1]), np.max(s[1])
        plt.fill([lam_0, lam_0, lam_1, lam_1],[y_min, y_max, y_max, y_min], alpha=0.3, c='b', label='Region integrated for left plot')
        if save_fig:
            plt.savefig(save_nam)
            plt.close()


    def plot_spec_prof(self, pos_0='default', pos_1='default', lam_0='default', lam_1='default',  title_prof='default', title_spec='default', save_fig=False, save_nam='./', lognorm_prof=False,  lognorm_spec=False, bin_prof=False, bin_spec=False):
        if lam_0 == 'default':
            lam_0 = np.min(self.lam)
        if lam_1 == 'default':
            lam_1 = np.max(self.lam)
        if pos_0 == 'default':
            pos_0 = np.min(self.pos)
        if pos_1 == 'default':
            pos_1 = np.max(self.pos)
        plt.figure(figsize=(21,5))
        plt.subplot(122)
        self.plot_prof(lam_0=lam_0, lam_1=lam_1, title=title_prof, lognorm=lognorm_prof, bin=bin_prof)
        s = self.make_prof(lam_0=lam_0, lam_1=lam_1, bin=bin_prof)
        y_min, y_max = np.min(s[1]), np.max(s[1])
        plt.fill([pos_0, pos_0, pos_1, pos_1],[y_min, y_max, y_max, y_min], alpha=0.3, c='b', label='Region integrated for left plot')
        plt.subplot(121)
        self.plot_spec(pos_0=pos_0, pos_1=pos_1, title=title_spec, lognorm=lognorm_spec, bin=bin_spec)
        if save_fig:
            plt.savefig(save_nam)
            plt.close()

#%%

    def mask_region(self, lam_0='default', lam_1='default', pos_0='default', pos_1='default', unmask=False):
        if lam_0 == 'default':
            lam_0 = self.lam[0]
        if lam_1 == 'default':
            lam_1 = self.lam[-1]
        if pos_0 == 'default':
            pos_0 = self.pos[0]
        if pos_1 == 'default':
            pos_1 = self.pos[-1]
        if unmask:
            self.mask[self.ltx(lam_0):self.ltx(lam_1)+1,self.ptx(pos_0):self.ptx(pos_1)]=0
            self.obj.mask[self.ltx(lam_0):self.ltx(lam_1)+1,self.ptx(pos_0):self.ptx(pos_1)]=False
        else:
            self.mask[self.ltx(lam_0):self.ltx(lam_1)+1,self.ptx(pos_0):self.ptx(pos_1)]=1
            self.obj.mask[self.ltx(lam_0):self.ltx(lam_1)+1,self.ptx(pos_0):self.ptx(pos_1)]=True
        new_mask=copy(self.mask)
        self.update_mask(new_mask)

#%%
    def raw_cont_1D(self, spec, std='default', lambdas=False, deg=4):
        if std is 'default':
            if not lambdas:
                std = np.ones(np.shape(spec))
            else:
                std = np.ones(np.shape(lambdas))
        if lambdas is False:
            pol, res, osef1, osef2, osef3 = np.polyfit(self.lam, spec, deg, w=1/std, full=True)
            cont=np.poly1d(pol)(self.lam)
            std_per_pix=(res**0.5)/len(self.lam)
        else:
            pixs=[]
            for l in lambdas:
                pixs.append(self.ltx(l))
            pol, res, osef1, osef2, osef3 = np.polyfit(lambdas, spec[pixs], deg, full=True)
            cont=np.poly1d(pol)(self.lam)
            std_per_pix=(res/len(lambdas))**0.5

        std_c = np.ones(np.shape(cont))*std_per_pix

        return cont, std_c

    def plot_raw_cont_1D(self, pos, lambdas=False, std_data=True, fig=6, save_fig=False, save_nam='./', deg=4):
        plt.figure(fig)
        spec = self.obj[:, self.ptx(pos)]
        std = self.std[:, self.ptx(pos)]
        m = np.max(spec.data)
        plt.plot(self.lam, self.obj.data[:, self.ptx(pos)]/m, alpha=0.6, color='k', label='Spectrum', linestyle='--')
        cont, std_cont= self.raw_cont_1D(spec, std=std, lambdas=lambdas, deg=deg)
        plt.plot(self.lam, cont/m, label='Adjusted continuum')
        plt.fill_between(self.lam, (cont-3*std_cont)/m, (cont+3*std_cont)/m, color='blue', alpha=0.3, label="Uncertainty on continuum")
        if std_data:
            plt.fill_between(self.lam, (spec+3*std)/m, (spec-3*std)/m, color='r', alpha=0.3, label='Uncertainty on measurments')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux (Arbitrary unit)')
        pixs=[]
        for l in lambdas:
            pixs.append(self.ltx(l))
        plt.scatter(lambdas,spec[pixs]/m, color='green',label='Measurments')
        plt.legend()
        plt.title('Spectrum adjustment at '+str(pos)+' arcsecond from the photocenter')
        if save_fig:
            plt.savefig(save_nam)
            plt.close(fig)

    def plot_raw_cont_1D_int(self, pos_0, pos_1, lambdas=False, std_data=True, fig=6, save_fig=False, save_nam='./', deg=4, ret=True):
        plt.figure(fig)
        l, spec, std = self.make_spec(pos_0, pos_1)
        m = np.max(spec.data)*0+1
        plt.plot(self.lam, np.sum(self.obj.data[:, self.ptx(pos_0):self.ptx(pos_1)],1)*self.pix_scale/m, alpha=0.6, color='k', label='Spectrum', linestyle='--')
        cont, std_cont= self.raw_cont_1D(spec, std=std, lambdas=lambdas, deg=deg)
        plt.plot(self.lam, cont/m, label='Adjusted continuum')
        plt.fill_between(self.lam, (cont-3*std_cont)/m, (cont+3*std_cont)/m, color='blue', alpha=0.3, label="Uncertainty on continuum")
        if std_data:
            plt.fill_between(self.lam, (spec+3*std)/m, (spec-3*std)/m, color='r', alpha=0.3, label='Uncertainty on measurments')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux ($W.m^{-2}.\mu m^{-1}$)')
        pixs=[]
        for l in lambdas:
            pixs.append(self.ltx(l))
        plt.scatter(lambdas,spec[pixs]/m, color='green',label='Measurments')
        plt.legend()
        plt.title('Spectrum adjustment from '+str(pos_0)+' to '+str(pos_1)+' arcsecond from the photocenter')
        if save_fig:
            plt.savefig(save_nam)
            plt.close(fig)
        if ret:
            return cont/m


    def raw_cont_2D(self, lambdas=False, deg=4):
        cont=np.zeros(np.shape(self.obj))
        cont_std=np.zeros(np.shape(self.obj))
        for i in range(len(self.obj[0,:])):
            cont[:,i], cont_std[:,i] = self.raw_cont_1D(self.obj[:,i], std=self.std[:,i],lambdas=lambdas, deg=deg)
        return cont, cont_std

    def make_poly_cont_from_not_masked_regions(self, deg=4):
        cont=np.zeros(np.shape(self.obj))
        cont_std=np.zeros(np.shape(self.obj))
        for i in range(len(self.obj[0,:])):
            cont[:,i], cont_std[:,i] = self.raw_cont_1D(self.obj[:,i], std=self.std[:,i],lambdas=self.lam[np.where(self.obj.mask[:,i] == False)], deg=deg)
        return cont, cont_std

    def make_cont_2(self):
        self.cont = copy(self.obj.data)
        self.cont = np.reshape(scipy.interpolate.griddata(np.where(self.obj.mask == False), self.obj[np.where(self.obj.mask == False)], np.where(self.mask > -10), method='linear'), np.shape(self.obj.mask))
        for i in range(len(self.cont[0,:])):
            self.cont[:,i] = np.reshape(scipy.interpolate.griddata(np.where(np.isfinite(self.cont[:,i])), self.cont[:,i][np.where(np.isfinite(self.cont[:,i]))], np.where(self.mask[:,i] > -10), method='nearest'), np.shape(self.obj.mask[:,i]))
        return self.cont

    def def_cont(self, cont, std='default'):
        if std is 'default':
            std = np.zeros(np.shape(cont))
        self.cont = cont
        self.cont_std = std

    def def_raies_from_cont(self, cont, std='default'):
        if std is 'default':
            std = np.zeros(np.shape(cont))
        self.raies = self.obj.data-cont
        self.raies_std = (self.std.data**2+std**2)**0.5

    def make_cont_obj(self):
        return Continuum(self.cont, self.cont_std, self.lam, self.pos)


    def make_line_obj(self,lam_0,lam_1, name='Unknown', wvl='default', dble= False, abs=False):
        return Line(self.obj[self.ltx(lam_0):self.ltx(lam_1)],self.std[self.ltx(lam_0):self.ltx(lam_1)],self.lam[self.ltx(lam_0):self.ltx(lam_1)],self.pos, name=name, wvl=wvl, dble=dble, abs=abs)

    def make_lines_obj(self):
        return Lines(self.raies, self.raies_std, self.lam, self.pos)



#%%

class Spectre_1D:
    def __init__(self,lambdas,flux,header=None,typ=None,met=None,temp=None):
        self.lambdas=lambdas
        self.flux=flux
        self.header=header
        self.met=met
        self.typ=typ



class Continuum(Spec_Obj):
    def __init__(self, obj_final, std_obj_final, lam, pos):
        self.obj_nm = obj_final
        self.std_nm = std_obj_final
        self.mask = np.zeros(np.shape(self.obj_nm))
        self.obj = ma.masked_array(obj_final, self.mask)
        self.std = ma.masked_array(std_obj_final, self.mask)
        self.lam = lam
        self.pos = pos
        self.pix_scale = np.abs(np.mean((np.roll(self.pos,1)-self.pos)[2:-2]))
        self.pos = pos
        self.lam_pos = np.meshgrid(self.lam, self.pos)
        self.x = np.arange(len(self.obj_nm[:,0]))
        self.y = np.arange(len(self.obj_nm[0,:]))
        self.xy = np.meshgrid(self.x, self.y)
        self.pix_scale = np.abs(np.mean((np.roll(self.pos,1)-self.pos)[2:-2]))
        self.pix_bw = np.abs(np.mean((np.roll(self.lam,1)-self.lam)[2:-2]))
        self.specs = {}

    def load_ssps(self, rep='../SSP/'):
        file = sorted(glob.glob(rep+'*'))
        spec_th_temp = np.genfromtxt(file[0],skip_header=1)


        for i in range(2600):
            wave_ni=spec_th_temp[1:,1]/10
            flux_ni=spec_th_temp[1:,i]
            flux=np.interp(self.lam,wave_ni,flux_ni)
            flux /= np.mean(flux)
            self.specs['SSP-'+str(4*i)]=Spectre_1D(self.lam,flux)

    def load_bbs(self, T_min = 200, T_max = 40050, step = 5):
        for temp in range(T_min, T_max, step):
            bb = bbrad(self.lam/10**9,temp)
            bb /= np.mean(bb)
            self.specs['BB'+str(temp)]=Spectre_1D(self.lam, bb, typ='Black Body at '+str(temp)+' K',temp=temp)

    def load_pls(self, e_min = -400, e_max = 400, step = 1):
        for a in np.arange(e_min, e_max, step):
            pl = (self.lam**(a/100))/np.mean((self.lam**(a/100)))
            pl /= np.mean(pl)
            self.specs['PL'+str(a)]=Spectre_1D(self.lam, pl, typ='Power law with coeff '+str(a))

    def load_spec(self, spec, name):
        self.specs[name] = Spectre_1D(self.lam, spec, typ='Perso')

    def load_pickles(self, rep='/home/pvermot/Documents/2017/SPHERE/LSS/TRAITEMENT/DONNEES/SPEC_REF/'):

        # Using readlines()
        file1 = open(rep+'/dico.ascii', 'r')
        Lines = file1.readlines()

        # Strips the newline character
        for line in Lines:
            file, name, temp = line.split(',')
            lsp, sp = np.genfromtxt(rep+'/'+file+'.ascii', delimiter='  ').T
            sp_i = np.interp(self.lam, lsp/10., sp)
            self.load_spec(sp_i, name)

    def load_all(self):
        self.load_ssps()
        self.load_bbs()
        self.load_pls()
        self.load_pickles()


    def def_str_fonc(*args):
        farg=[]
        str_def='def f(lambdas,'
        str_arg='arg=['
        str_ret='return '
        for i in range(len(args)):
            farg.append('c'+str(i))
            if i!=0:
                str_def+=',c'+str(i)
                str_ret+='+c'+str(i)+'*specs[arg['+str(i)+']].flux'
                str_arg+=','+"'"+args[i]+"'"
            else:
                str_def+='c'+str(i)
                str_arg+="'"+args[i]+"'"
                str_ret+='c'+str(i)+'*specs[arg['+str(i)+']].flux'
        str_def+='):'
        str_arg+=']'
        return str_def+'\n \t'+str_arg+'\n \t'+str_ret



    def fit_conti(self, cont1D, spes, err_1D=False, std=False, p0=0, bounds=0, plot=False, retour='default', maxfev=1e7):
        spectres=[]
        for spectre in spes:
            spectres.append(self.specs[spectre].flux)
        if p0==0:
            p0=[]
            for sp in spectres:
                p0.append(np.sum(cont1D)/(len(spectres)*np.sum(sp)))
        if bounds==0:
            bounds=(np.full(np.shape(spes),0),np.full(np.shape(spes),np.inf))

        def wrapper(lam, *args): #take a list of arguments and break it down into two lists for the fit function to understand
            N = len(args)
            amplitudes = list(args[0:N])
            return fit_func(lam, amplitudes)

        def fit_func(lam, amplitudes): #the actual fit function
            fit = np.zeros(len(lam))
            j=0
            for m in amplitudes:
                fit += abs(m)*spectres[j]
                j+=1
            return fit
#        print(p0)
        try:
            popt, pcov = curve_fit(lambda x, *p0: wrapper(self.lam, *p0), self.lam, cont1D, p0=p0,maxfev=int(maxfev)) #call with lambda function
            perr = np.sqrt(np.diag(pcov))
        except:
            popt = p0
            perr = p0
#        perr = np.sqrt(np.diag(pcov))
        fit = np.zeros(np.shape(self.lam))
        fit_var = np.zeros(np.shape(self.lam))
        for a in range(len(popt)):
            fit+=spectres[a]*abs(popt[a])
            fit_var += (spectres[a]*perr[a])**2
        fit_std = fit_var**0.5
        if plot:
            if not err_1D is False:
                plt.fill_between(self.lam, cont1D+err_1D, cont1D-err_1D, color='k', alpha=0.3, label='Uncertainty on continuum')
            plt.plot(self.lam,cont1D,label='Continuum',color='k')
            for a in range(len(popt)):
                pt = plt.plot(self.lam,spectres[a]*abs(popt[a]),label=spes[a])
                if not std is False:
                    plt.fill_between(self.lam, spectres[a]*(abs(popt[a])+3*perr[a]), spectres[a]*(abs(popt[a])-3*perr[a]), color=pt[0].get_color(), alpha=0.3, label=r'$\Delta$'+spes[a])
            pt = plt.plot(self.lam,fit,label='Result of the fit')
            if not std is False:
                plt.fill_between(self.lam, fit+3*fit_std, fit-3*fit_std, color=pt[0].get_color(), alpha=0.3, label='Uncertainty on the fit')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel(r"Flux $(W.m^{-2}.\mu m^{-1})$")
            plt.legend(prop={'size': 8})
        res = np.sum(np.abs(cont1D-fit))
        fit_std
        if retour is 'fit':
#            print('Retour fit')
            return fit
        if retour is 'default':
            return np.abs(popt), perr
        if retour is 'res':
            return np.abs(popt), res


    def fit_plot(self, pos_0, pos_1, spes, std=False, err=False):
        lam, cont, err = self.make_spec(pos_0,pos_1)
        self.fit_conti(cont, spes, err_1D=err, plot=True, std=std)

    def get_fit(self, pos_0, pos_1, spes, plot=False):
        lam, cont, std = self.make_spec(pos_0,pos_1)
        fit =  self.fit_conti(cont, spes, err_1D=std, plot=plot, retour='fit')
        return fit

    def load_fit(self, pos_0, pos_1, spes, name='fit', plot=False):
        fit = self.get_fit(pos_0, pos_1, spes, plot=plot)
        self.load_spec(fit, name)

    def fit_props(self, cont1D, spes, std=False):
        if std is False:
            param, res = self.fit_conti(cont1D,spes, retour='res')
            spectres=[]
            for spectre in spes:
                spectres.append(self.specs[spectre].flux)
            p=np.array(param)
            props=np.zeros(np.shape(p))
            for pa in range(len(p)):
                props[pa]=np.sum(spectres[pa])*p[pa]/np.sum(cont1D)
            return props, res
        else:
            param, err = self.fit_conti(cont1D,spes)
            spectres=[]
            for spectre in spes:
                spectres.append(self.specs[spectre].flux)
            p=np.array(param)
            props=np.zeros(np.shape(p))
            errs=np.zeros(np.shape(p))
            for pa in range(len(p)):
                props[pa]=np.sum(spectres[pa])*p[pa]/np.sum(cont1D)
                errs[pa]=np.sum(spectres[pa])*err[pa]/np.sum(cont1D)
            return props, errs

    def plot_fluxs(self, spec2D, spe, lognorm=False, mean=1, n=False, errs=True, offset=False, offset_label='Background'):
        if errs == True:
            props = []
            posi = []
            ress = []
            errs=[]
            for i in range(1,len(self.obj[0,:]),mean):
                pros, ers = self.fit_props(np.mean(spec2D[:,i:i+mean],1),spe, std=True)
                maxit=np.sum(np.mean(spec2D[:,i:i+mean],1))
                props.append(np.array(pros)*np.sum(np.mean(spec2D[:,i:i+mean],1)))
                errs.append(np.array(ers)*np.sum(np.mean(spec2D[:,i:i+mean],1)))
                posi.append(np.mean(self.pos[i:i+mean]))
                ress.append(self.fit_props(np.mean(spec2D[:,i:i+mean],1),spe)[1])
            ps=np.array(props)
            es=np.array(errs)
            maxi=0
            plt.plot(0,0)
            for i in range(1,len(self.obj[0,:]),mean):
                maxit=np.sum(np.mean(spec2D[:,i:i+mean],1))
                if maxit>maxi:
                    maxi=maxit
            for j in range(len(ps[0,:])):
                if n is False:
                    plot = plt.step(posi,ps[:,j]/maxi,label=spe[j], where='mid')
                    plt.fill_between(posi,(ps[:,j]+3*es[:,j])/maxi,(ps[:,j]-3*es[:,j])/maxi,label=spe[j], step='mid', alpha=0.3, color=plot[0].get_c())
                else:
                    if j is n:
                        plt.step(posi,ps[:,j]/maxi,label=spe[j])
            if offset != False:
                plt.plot(posi, np.array(posi)*0+offset, label=offset_label)
            if lognorm == 'x':
                plt.semilogx()
            if lognorm == 'y':
                plt.semilogy()
            if lognorm == 'xy':
                plt.loglog()
            plt.xlabel('Relative position to the photocenter (arcsecond)')
            plt.ylabel('Relative flux of each component')
            plt.legend()
        else:
            props = []
            posi = []
            ress = []
            for i in range(1,len(self.obj[0,:]),mean):
                pros, ers = self.fit_props(np.mean(spec2D[:,i:i+mean],1),spe, std=True)
                props.append(np.array(pros)*np.sum(np.mean(spec2D[:,i:i+mean],1)))
                posi.append(np.mean(self.pos[i:i+mean]))
                ress.append(self.fit_props(np.mean(spec2D[:,i:i+mean],1),spe)[1])
            ps=np.array(props)
            print(np.shape(ps))
            maxi=0
            plt.plot(0,0)
            for i in range(1,len(self.obj[0,:]),mean):
                maxit=np.sum(np.mean(spec2D[:,i:i+mean],1))
                if maxit>maxi:
                    maxi=maxit
            for j in range(len(ps[0,:])):
                if n is False:
                    plt.step(posi,ps[:,j]/maxi,label=spe[j], where='mid')
                else:
                    if j is n:
                        plt.step(posi,ps[:,j]/maxi,label=spe[j])
            if offset != False:
                plt.plot(posi, np.array(posi)*0+offset, label=offset_label)
            if lognorm == 'x':
                plt.semilogx()
            if lognorm == 'y':
                plt.semilogy()
            if lognorm == 'xy':
                plt.loglog()
            plt.xlabel('Relative position to the photocenter (arcsecond)')
            plt.ylabel('Relative flux of each component')
            plt.legend()
        return posi, ps


    def plot_fluxs_res(self, spec2D, spe, mean=1, plot=True):
        props = []
        posi = []
        ress = []
        for i in range(1,len(spec2D[0,:]),mean):
            props.append(np.array(self.fit_props(np.mean(spec2D[:,i:i+mean],1),spe)[0])*np.sum(np.mean(spec2D[:,i:i+mean],1)))
            posi.append(np.mean(self.pos[i:i+mean]))
            ress.append(self.fit_props(np.mean(spec2D[:,i:i+mean],1),spe)[1])
        ps=np.array(props)
#        print(np.shape(ps))
        maxi=0
        for i in range(1,len(self.obj[0,:]),mean):
            maxit=np.sum(np.mean(spec2D[:,i:i+mean],1))
            if maxit>maxi:
                maxi=maxit
        if plot:
            plt.step(posi,ress)
            plt.xlabel('Relative position to the photocenter (arcsecond)')
            plt.ylabel('Residuals of the fit')
            plt.legend()
        return np.sum(ress)

#%%

class Lines(Spec_Obj):
    def __init__(self, obj_final, std_obj_final, lam, pos):
        self.obj_nm = obj_final
        self.std_nm = std_obj_final
        self.mask = np.zeros(np.shape(self.obj_nm))
        self.obj = ma.masked_array(obj_final, self.mask)
        self.std = ma.masked_array(std_obj_final, self.mask)
        self.lam = lam
        self.pos = pos
        self.pix_scale = np.abs(np.mean((np.roll(self.pos,1)-self.pos)[2:-2]))
        self.pos = pos
        self.lam_pos = np.meshgrid(self.lam, self.pos)
        self.x = np.arange(len(self.obj_nm[:,0]))
        self.y = np.arange(len(self.obj_nm[0,:]))
        self.xy = np.meshgrid(self.x, self.y)
        self.pix_scale = np.abs(np.mean((np.roll(self.pos,1)-self.pos)[2:-2]))
        self.pix_bw = np.abs(np.mean((np.roll(self.lam,1)-self.lam)[2:-2]))
    def a():
        return 42

#%%

class Line(Spec_Obj):
    def __init__(self, obj_final, std_obj_final, lam, pos, name='Unknown', wvl='default', dble= False, abs=False):
        self.obj_nm = obj_final
        self.std_nm = std_obj_final
        self.mask = np.zeros(np.shape(self.obj_nm))
        self.obj = ma.masked_array(obj_final, self.mask)
        self.std = ma.masked_array(std_obj_final, self.mask)
        self.lam = lam
        self.pos = pos
        self.pix_scale = np.abs(np.mean((np.roll(self.pos,1)-self.pos)[2:-2]))
        self.pos = pos
        self.lam_pos = np.meshgrid(self.lam, self.pos)
        self.x = np.arange(len(self.obj_nm[:,0]))
        self.y = np.arange(len(self.obj_nm[0,:]))
        self.xy = np.meshgrid(self.x, self.y)
        self.pix_scale = np.abs(np.mean((np.roll(self.pos,1)-self.pos)[2:-2]))
        self.pix_bw = np.abs(np.mean((np.roll(self.lam,1)-self.lam)[2:-2]))

        if wvl is 'default':
            self.wvl = np.mean(self.lam)
        else:
            self.wvl = wvl
            self.wvl_ref = wvl

        if name is 'Unknown':
            self.name = str(self.wvl)
        else:
            self.name = name

        if dble is False:
            self.dble = self.wvl
        else:
            self.dble = dble

#%%

    def fit_gauss_simple(self, lam, spec, sigma='default'):
        med = np.median(spec)
        mini = np.min(spec)
        maxi = np.max(spec)
        if abs(maxi-med)>abs(mini-med):
                p1 = maxi-med
        else:
                p1 = mini-med
        p0 = [self.wvl, p1, 0.03, 0, med]
        try:
            if sigma is 'default':
                param, cov = curve_fit(gauss, lam, spec, p0 = p0)
            else:
                param, cov = curve_fit(gauss, lam, spec, p0 = p0, sigma=sigma)
            flag=True
        except:
            param = p0
            cov = np.zeros((len(param),len(param)))
            flag = False
        return param, np.sqrt(np.diag(cov)), flag

    def fit_gauss_db_free(self, lam, spec, sigma='default'):
        p0 = [self.wvl, self.dble, np.max(spec)/2, np.max(spec)/2, 0.03, 0, 0]
        try:
            if sigma is 'default':
                param, cov = curve_fit(db_gauss, lam, spec, p0 = p0)
            else:
                param, cov = curve_fit(db_gauss, lam, spec, p0 = p0, sigma=sigma)
            flag=True
        except:
            param = p0
            cov = np.zeros((len(param),len(param)))
            flag = False
        return param, np.sqrt(np.diag(cov)), flag

    def fit_gauss_db_cons(self, lam, spec, sigma='default'):
        def temp_gauss(x, amp_0, amp_1, alpha, b):
            return amp_0*np.exp(-alpha*(x-self.wvl)**2)+amp_1*np.exp(-alpha*(x-self.dble)**2)+b
        p0 = [spec[self.ltx(self.wvl)]/2, spec[self.ltx(self.dble)]/2, 0.03,0]
        try:
            if sigma is 'default':
                param, cov = curve_fit(temp_gauss, lam, spec, p0 = p0)
            else:
                param, cov = curve_fit(temp_gauss, lam, spec, p0 = p0, sigma=sigma)
            flag=True
        except:
            param = p0
            cov = np.zeros((len(param),len(param)))
            flag = False
        return param, np.sqrt(np.diag(cov)), flag

    def fit_gauss_db_cons_pos(self, lam, spec, sigma='default'):
        def temp_gauss(x, amp_0, amp_1, alpha, b):
            if (amp_0>0) and (amp_1>0):
                return amp_0*np.exp(-alpha*(x-self.wvl)**2)+amp_1*np.exp(-alpha*(x-self.dble)**2)+b
            else:
                return x*1e6
        p0 = [spec[self.ltx(self.wvl)]/2, spec[self.ltx(self.dble)]/2, 0.03,0]
        try:
            if sigma is 'default':
                param, cov = curve_fit(temp_gauss, lam, spec, p0 = p0)
            else:
                param, cov = curve_fit(temp_gauss, lam, spec, p0 = p0, sigma=sigma)
            flag=True
        except:
            param = p0
            cov = np.zeros((len(param),len(param)))
            flag = False
        return param, np.sqrt(np.diag(cov)), flag


    def fit_gauss_db_half_cons(self, lam, spec, sigma='default'):
        def temp_gauss(x, x_0, amp_0, amp_1, alpha):
            return amp_0*np.exp(-alpha*(x-x_0)**2)+amp_1*np.exp(-alpha*(x-self.dble)**2)
        p0 = [self.wvl, np.max(spec)/2, np.max(spec)/2, 0.03]
        try:
            if sigma is 'default':
                param, cov = curve_fit(temp_gauss, lam, spec, p0 = p0)
            else:
                param, cov = curve_fit(temp_gauss, lam, spec, p0 = p0, sigma=sigma)
            flag=True
        except:
            param = p0
            cov = np.zeros((len(param),len(param)))
            flag = False
        return param, np.sqrt(np.diag(cov)), flag

    def fit_gauss(self, lam, spec, sigma='default', mode='simple'):
        if mode is 'simple':
            return self.fit_gauss_simple(lam, spec, sigma=sigma)
        if mode is 'db_free':
            return self.fit_gauss_db_free(lam, spec, sigma=sigma)
        if mode is 'db_cons':
            return self.fit_gauss_db_cons(lam, spec, sigma=sigma)
        if mode is 'db_cons_pos':
            return self.fit_gauss_db_cons_pos(lam, spec, sigma=sigma)
        if mode is 'db_half_cons':
            return self.fit_gauss_db_half_cons(lam, spec, sigma=sigma)


#%%

    def a2std(self, a, err=False):
         if err is False:
             return (1/(2*a))**0.5
         else:
             return (1/(2*a))**0.5, err*(1/(a*2))**1.5

    def a2fwhm(self, a, err=False):
        if err is False:
            return 2.355*self.a2std(a, err)
        else:
            return 2.355*self.a2std(a, err)[0] , 2.355*self.a2std(a, err)[1]

    def ampa2fl(self, amp, a, err_amp=False, err_a=False):
        if (err_amp is False) or (err_a is False):
            return amp*(np.pi/a)**0.5
        else:
            return amp*(np.pi/a)**0.5, ((err_amp**(np.pi/a)**0.5)**2 + (amp*np.pi**0.5*err_a/a**2)**2)**0.5

#%%

    def fit(self, pos_0, pos_1, mode='simple'):
        lam, spec, std = self.make_spec(pos_0=pos_0, pos_1=pos_1)
        param, std, flag = self.fit_gauss(lam, spec, sigma=std, mode=mode)
        return param, std, flag


#%%

    def fit_all_pos2(self, step=False, bin=False, mode='simple'):
        specs = []
        stds = []
        poss = []
        bin = copy(step)
        step = False

        if step is False:
            for i in range(len(self.pos)-1):
                l, o, s = self.make_spec(self.pos[i],self.pos[i+1])
                specs.append(o)
                stds.append(s)
                poss.append([self.pos[i],self.pos[i+1]])
                lambdas=l
        else:
            p0 = step*int((self.pos[0]+step/2)/step)-step/2
            p1 = step*int((self.pos[-1]-step/2)/step)+step/2
            for pos_0 in np.arange(p0, p1, step):
                l, o, s = self.make_spec(pos_0, pos_0+step)
                specs.append(o)
                stds.append(s)
                poss.append([pos_0,pos_0+step])

        params, errs, posis, flags = [], [], [], []
        for i in range(len(poss)):
            p, c ,f = self.fit(poss[i][0], poss[i][1], mode=mode)
            posis.append((poss[i][0]+poss[i][1])/2)
            flags.append(f)
            errs.append(c)
            params.append(p)

        bin_fac=int(bin/self.pix_scale)
        params_b = np.zeros((int(np.shape(np.array(params))[0]/bin_fac),np.shape(np.array(params))[1]))
        errs_b = np.zeros((int(np.shape(np.array(params))[0]/bin_fac),np.shape(np.array(params))[1]))
        for i in range(len(params[0])):
            print(bin, step, bin_fac, np.shape(np.array(params)),np.shape(np.array(params_b)),len(params[0]))
            params_b[:,i] = rebin(np.array(params)[:,i], bin_fac, mode='mean')
            errs_b[:,i] = rebin(np.array(errs)[:,i], bin_fac, mode='std_mean')
        posis=rebin(np.array(posis), bin_fac, mode='mean')
        flags=rebin(np.array(flags), bin_fac, mode='mean')
        flags=flags*0+1


        if mode is 'simple':
            return posis, np.array([params_b[:,0], params_b[:,1], params_b[:,2]]), np.array([errs_b[:,0], errs_b[:,1], errs_b[:,2]]), flags
        if mode is 'db_free':
            return posis, np.array([params_b[:,0], params_b[:,2], params_b[:,4]]), np.array([errs_b[:,0], errs_b[:,2], errs_b[:,4]]), flags
        if mode is 'db_cons':
            return posis, np.array([params_b[:,0], params_b[:,2]]), np.array([errs_b[:,0], errs_b[:,2]]), flags
        if mode is 'db_cons_pos':
            return posis, np.array([params_b[:,0], params_b[:,2]]), np.array([errs_b[:,0], errs_b[:,2]]), flags
        if mode is 'db_half_cons':
            return posis, np.array([params_b[:,0], params_b[:,1], params_b[:,3]]), np.array([errs_b[:,0], errs_b[:,1], errs_b[:,3]]), flags

#%%

    def fit_all_pos(self, step=False, bin=False, mode='simple'):
        specs = []
        stds = []
        poss = []
        if step is False:
            for i in range(len(self.pos)-1):
                l, o, s = self.make_spec(self.pos[i],self.pos[i+1])
                specs.append(o)
                stds.append(s)
                poss.append([self.pos[i],self.pos[i+1]])
                lambdas=l
        else:
            p0 = step*int((self.pos[0]+step/2)/step)-step/2
            p1 = step*int((self.pos[-1]-step/2)/step)+step/2
            for pos_0 in np.arange(p0, p1, step):
                l, o, s = self.make_spec(pos_0, pos_0+step)
                specs.append(o)
                stds.append(s)
                poss.append([pos_0,pos_0+step])

        params, errs, posis, flags = [], [], [], []
        for i in range(len(poss)):
            p, c ,f = self.fit(poss[i][0], poss[i][1], mode=mode)
            posis.append((poss[i][0]+poss[i][1])/2)
            flags.append(f)
            errs.append(c)
            params.append(p)

        params=np.array(params)
        errs=np.array(errs)
        posis=np.array(posis)
        flags=np.array(flags)


        if mode is 'simple':
            return posis, np.array([params[:,0], params[:,1], params[:,2]]), np.array([errs[:,0], errs[:,1], errs[:,2]]), flags
        if mode is 'db_free':
            return posis, np.array([params[:,0], params[:,2], params[:,4]]), np.array([errs[:,0], errs[:,2], errs[:,4]]), flags
        if mode is 'db_cons':
            return posis, np.array([params[:,0], params[:,2]]), np.array([errs[:,0], errs[:,2]]), flags
        if mode is 'db_cons_pos':
            return posis, np.array([params[:,0], params[:,2]]), np.array([errs[:,0], errs[:,2]]), flags
        if mode is 'db_half_cons':
            return posis, np.array([params[:,0], params[:,1], params[:,3]]), np.array([errs[:,0], errs[:,1], errs[:,3]]), flags


#%%
    def plot_fit_simple(self, pos_0, pos_1, save_fig=False, save_nam='./', lognorm=False, bin=False):
        self.plot_spec(pos_0, pos_1)
        p, c, f = self.fit(pos_0, pos_1,mode= 'simple')
        if f == True:
            plt.plot(self.lam, gauss(self.lam, *p), color='blue', label='Fit')
        plt.legend()
        print(p, c, f)

    def plot_fit_db_free(self, pos_0, pos_1, save_fig=False, save_nam='./', lognorm=False, bin=False):
        self.plot_spec(pos_0, pos_1)
        p, c, f = self.fit(pos_0, pos_1,mode= 'db_free')
        if f == True:
            plt.plot(self.lam,db_gauss(self.lam, *p), color='blue', label='Full fit')
            plt.plot(self.lam,gauss(self.lam, *[p[0], p[2], p[4], p[5], p[6]]), color='green', label='Fit line 1')
            plt.plot(self.lam,gauss(self.lam, *[p[1], p[3], p[4], p[5], p[6]]), color='yellow', label='Fit line 2')
        plt.legend()
        print(self.fit(pos_0, pos_1))

    def plot_fit_db_cons(self, pos_0, pos_1, save_fig=False, save_nam='./', lognorm=False, bin=False):
        self.plot_spec(pos_0, pos_1)
        p, c, f = self.fit(pos_0, pos_1,mode= 'db_cons')
        def temp_gauss(x, amp_0, amp_1, alpha, b):
            return amp_0*np.exp(-alpha*(x-self.wvl)**2)+amp_1*np.exp(-alpha*(x-self.dble)**2)+b
        if f == True:
            plt.plot(self.lam, temp_gauss(self.lam, *p), color='blue', label='Full fit')
            plt.plot(self.lam, gauss(self.lam, *[self.wvl, p[0], p[2], 0, 0]), color='green', label='Fit line 1')
            plt.plot(self.lam, gauss(self.lam, *[self.dble, p[1], p[2], 0, 0]), color='yellow', label='Fit line 2')
            plt.plot(self.lam, gauss(self.lam, *[0, 0, 0, 0, p[3]]), color='red', label='Continuum')
        plt.legend()
        print(self.fit(pos_0, pos_1,mode= 'db_cons'))

    def plot_fit_db_cons_pos(self, pos_0, pos_1, save_fig=False, save_nam='./', lognorm=False, bin=False):
        self.plot_spec(pos_0, pos_1)
        p, c, f = self.fit(pos_0, pos_1,mode= 'db_cons_pos')
        def temp_gauss(x, amp_0, amp_1, alpha, b):
            if (amp_0>0) and (amp_1>0):
                return amp_0*np.exp(-alpha*(x-self.wvl)**2)+amp_1*np.exp(-alpha*(x-self.dble)**2)+b
            else:
                return x*1e6
        if f == True:
            plt.plot(self.lam, temp_gauss(self.lam, *p), color='blue', label='Full fit')
            plt.plot(self.lam, gauss(self.lam, *[self.wvl, p[0], p[2], 0, 0]), color='green', label='Fit line 1')
            plt.plot(self.lam, gauss(self.lam, *[self.dble, p[1], p[2], 0, 0]), color='yellow', label='Fit line 2')
            plt.plot(self.lam, gauss(self.lam, *[0, 0, 0, 0, p[3]]), color='red', label='Continuum')
        plt.legend()
        print(self.fit(pos_0, pos_1,mode= 'db_cons_pos'))

    def plot_fit_db_half_cons(self, pos_0, pos_1, save_fig=False, save_nam='./', lognorm=False, bin=False):
        self.plot_spec(pos_0, pos_1)
        p, c, f = self.fit(pos_0, pos_1,mode= 'db_half_cons')
        def temp_gauss(x, x_0, amp_0, amp_1, alpha):
            return amp_0*np.exp(-alpha*(x-x_0)**2)+amp_1*np.exp(-alpha*(x-self.dble)**2)
        if f == True:
            plt.plot(self.lam, temp_gauss(self.lam, *p), color='blue', label='Full fit')
            plt.plot(self.lam, gauss(self.lam, *[p[0], p[1], p[3], 0, 0]), color='green', label='Fit line 1')
            plt.plot(self.lam, gauss(self.lam, *[self.dble, p[2], p[3], 0, 0]), color='yellow', label='Fit line 2')
        plt.legend()
        print(self.fit(pos_0, pos_1,mode= 'db_half_cons'))


    def plot_fit_gauss(self, pos_0, pos_1, mode='simple', save_fig=False, save_nam='./', lognorm=False, bin=False, title='default'):
        if mode is 'simple':
            self.plot_fit_simple( pos_0, pos_1)
        if mode is 'db_free':
            self.plot_fit_db_free( pos_0, pos_1)
        if mode is 'db_cons':
            self.plot_fit_db_cons( pos_0, pos_1)
        if mode is 'db_cons_pos':
            self.plot_fit_db_cons_pos( pos_0, pos_1)
        if mode is 'db_half_cons':
            self.plot_fit_db_half_cons( pos_0, pos_1)
        if lognorm == 'x':
            plt.semilogx()
        if lognorm == 'y':
            plt.semilogy()
        if lognorm == 'xy':
            plt.loglog()
        if not title is 'default':
            plt.title(title, fontsize=11, fontweight="bold")
        if save_fig:
            plt.savefig(save_nam)
            plt.close()


#%%

    def plot_pos(self, step=0.15, mode='simple', params=[3,3,5,5], hide_errors=True, speed=False, save_fig=False, save_nam='./', lognorm=False, bin=False, title='default', yaxis=2, ret=False):
        po, pa, err, fl1 = self.fit_all_pos(step=step, mode=mode)
        i = np.argmin(po**2)
        ref = np.median(pa[0, i-1:i+2])
        print(ref)
        p2a, p2b, p3, p4 = params[0], params[1], params[2],params[3]
        fl2 = (po>-p2a)*(po<p2b)
        fl3 = err[0,:]<p3
        where = np.where(fl1*fl2*fl3 == True)
        fl4 = (pa[0,:] > np.median(pa[0, where][0])-p4)*(pa[0,:] < np.median(pa[0, where][0])+p4)
        print(np.shape(po), np.shape(pa), po[np.where(fl1*fl2*fl3 == True)])
        where = np.where(fl1*fl2*fl3*fl4 == True)
        if not hide_errors:
            where = np.where(fl1*fl2*fl3*fl4 != 1048)
        self.wvl = np.median(pa[0, where][0])
        plt.xlabel('Position relative to photocenter (arcsecond)')
        err /= (step/self.pix_scale)**0.5
        if speed is True:
            cel =299792.458
            plt.plot([-5.91, 6.08], [0, 0], linestyle='--', label='Reference')
            plt.fill_between(po[where], cel*((pa[0, where][0]-3*err[0, where][0])-ref)/ref, y2=cel*((pa[0, where][0]+3*err[0, where][0])-ref)/ref, color='r', alpha=0.3)
            plt.scatter(po[where], cel*((pa[0, where])-ref)/ref)
            plt.axis([-5.91, 6.08, cel*((np.mean(pa[0, where][0])-yaxis-ref)/ref), cel*((np.mean(pa[0, where][0])+yaxis-ref)/ref)])
            plt.ylabel(r'Relative spectral position expressed in speed ($km.s^{-1}$)')
        else:
            plt.plot([-5.91, 6.08], [ref, ref], linestyle='--', label='Reference wavelength')
            plt.fill_between(po[where], (pa[0, where][0]-3*err[0, where][0]), y2=pa[0, where][0]+3*err[0, where][0], color='r', alpha=0.3)
            plt.scatter(po[where], (pa[0, where]))
            plt.ylabel(r'Spectral position (nm)')
            plt.axis([-5.91, 6.08, np.mean(pa[0, where][0])-yaxis, np.mean(pa[0, where][0])+yaxis])
        print(np.shape( pa[0, where]), err[0, where], 3*err[0, where])
        if lognorm == 'x':
            plt.semilogx()
        if lognorm == 'y':
            plt.semilogy()
        if lognorm == 'xy':
            plt.loglog()
        if title == 'default':
            title = 'Position of '+str(self.wvl)
        plt.title(title, fontsize=11, fontweight="bold")
        if save_fig:
            plt.savefig(save_nam)
            plt.close()
        if ret:
            if speed is True:
                return po[where], cel*((pa[0, where])-ref)/ref
            else:
                return po[where], (pa[0, where])

    def plot_std(self, step=0.15, mode='simple', params=[3,3,5,5], speed=False, save_fig=False, save_nam='./', lognorm=False, bin=False, title='default'):
        po, pa, err, fl1 = self.fit_all_pos(step=step, mode=mode)
        st, er = self.a2std(pa[2,:], err[2, :])
#        er = self.a2std(err[2,:])
        p2a, p2b, p3, p4 = params[0], params[1], params[2],params[3]
        fl2 = (po>-p2a)*(po<p2b)
        fl3 = er < p3
        where = np.where(fl1*fl2*fl3 == True)
        self.wid=np.median(st[where])
        fl4 = (st >self.wid-p4)*(st < self.wid+p4)
        print(np.shape(po), np.shape(st), po[np.where(fl1*fl2*fl3 == True)])
        where = np.where(fl1*fl2*fl3*fl4 == True)
        self.wvl = np.median(pa[0, where][0])
        plt.xlabel('Position relative to photocenter (arcsecond)')
        if speed is True:
            cel =299792.458
            plt.fill_between(po[where], cel*((st[where]-3*er[where]))/self.wvl, y2=cel*((st[where]+3*er[where]))/self.wvl, color='r', alpha=0.3)
            plt.scatter(po[where], cel*((st[where]))/self.wvl)
            plt.axis([-5.91, 6.08, cel*((self.wid-5)/self.wvl), cel*((self.wid+5)/self.wvl)])
            plt.ylabel(r'Line width expressed in speed ($km.s^{-1}$)')
        else:
            plt.fill_between(po[where], ((st[where]-3*er[where])), y2=((st[where]+3*er[where])), color='r', alpha=0.3)
            plt.scatter(po[where], ((st[where])))
            plt.axis([-5.91, 6.08, (self.wid-5), ((self.wid+5))])
            plt.ylabel(r'Line width, std (nm)')
        print(st, er)
        if lognorm == 'x':
            plt.semilogx()
        if lognorm == 'y':
            plt.semilogy()
        if lognorm == 'xy':
            plt.loglog()
        if title == 'default':
            title = 'Std of '+str(self.wvl)
        plt.title(title, fontsize=11, fontweight="bold")
        if save_fig:
            plt.savefig(save_nam)
            plt.close()

    def plot_fwhm(self, step=0.15, mode='simple', params=[3,3,5,5], speed=False, save_fig=False, save_nam='./', lognorm=False, bin=False, title='default'):
        po, pa, err, fl1 = self.fit_all_pos(step=step, mode=mode)
        st, er = self.a2fwhm(pa[2,:], err[2, :])
#        er = self.a2std(err[2,:])
        p2a, p2b, p3, p4 = params[0], params[1], params[2],params[3]
        fl2 = (po>-p2a)*(po<p2b)
        fl3 = er < p3
        where = np.where(fl1*fl2*fl3 == True)
        self.wid=np.median(st[where])
        fl4 = (st >self.wid-p4)*(st < self.wid+p4)
        print(np.shape(po), np.shape(st), po[np.where(fl1*fl2*fl3 == True)])
        where = np.where(fl1*fl2*fl3*fl4 == True)
        self.wvl = np.median(pa[0, where][0])
        plt.xlabel('Position relative to photocenter (arcsecond)')
        if speed is True:
            cel =299792.458
            plt.fill_between(po[where], cel*((st[where]-3*er[where]))/self.wvl, y2=cel*((st[where]+3*er[where]))/self.wvl, color='r', alpha=0.3)
            plt.scatter(po[where], cel*((st[where]))/self.wvl)
            plt.axis([-5.91, 6.08, cel*((self.wid-5)/self.wvl), cel*((self.wid+5)/self.wvl)])
            plt.ylabel(r'Line width expressed in speed ($km.s^{-1}$)')
        else:
            plt.fill_between(po[where], ((st[where]-3*er[where])), y2=((st[where]+3*er[where])), color='r', alpha=0.3)
            plt.scatter(po[where], ((st[where])))
            plt.axis([-5.91, 6.08, (self.wid-5), ((self.wid+5))])
            plt.ylabel(r'Line width, FWHM (nm)')
        print(st, er)
        if lognorm == 'x':
            plt.semilogx()
        if lognorm == 'y':
            plt.semilogy()
        if lognorm == 'xy':
            plt.loglog()
        if title == 'default':
            fwhm = 'Std of '+str(self.wvl)
        plt.title(title, fontsize=11, fontweight="bold")
        if save_fig:
            plt.savefig(save_nam)
            plt.close()


    def plot_prof_fit(self, step=0.15, mode='simple', params=[3,3,5,5], save_fig=False, save_nam='./', lognorm=False, bin=False, title='default'):
        po, pa, err, fl1 = self.fit_all_pos(step=step, mode=mode)
        if (mode is 'db_cons') or (mode is 'db_cons_pos'):
            fl, er = self.ampa2fl(pa[0,:], pa[1,:], err_amp=err[0,:], err_a=err[1,:])
            if (mode is 'db_cons'):
                er /= 3
        else:
            fl, er = self.ampa2fl(pa[1,:], pa[2,:], err_amp=err[1,:], err_a=err[2,:])
        fl *= self.pix_bw/(abs(po[1]-po[0]))
        er *= self.pix_bw/(abs(po[1]-po[0]))
        print(fl)
        p2a, p2b, p3, p4 = params[0], params[1], params[2],params[3]
        fl2 = (po>-p2a)*(po<p2b)
        fl3 = er < p3
        where = np.where(fl1*fl2*fl3 == True)
        print(po[where])
        plt.fill_between(po[where], fl[where]-er[where], y2=fl[where]+er[where], color='r', alpha=0.3, step='mid')
        plt.step(po[where], fl[where], where='mid', color='k')
        print(np.min(fl[where]), np.max(fl[where]))
        plt.axis([-5.91, 6.08, np.min(fl[where])-0.1*abs(np.min(fl[where])), 1.1*np.max(fl[where])])
        plt.xlabel('Position relative to photocenter (arcsecond)')
        plt.ylabel('Flux ($W.m^{-2}.arcsecond^{-1}$)', fontsize=11)
        if lognorm == 'x':
            plt.semilogx()
        if lognorm == 'y':
            plt.semilogy()
        if lognorm == 'xy':
            plt.loglog()
        if title == 'default':
            title = 'Flux from gaussian adjustment of '+str(self.wvl)
        plt.title(title, fontsize=11, fontweight="bold")
        if save_fig:
            plt.savefig(save_nam)
            plt.close()
        return po, fl, er, where

    def plot_ew_fit(self, conti, step=0.15, mode='simple', params=[3,3,5,5], save_fig=False, save_nam='./', lognorm=False, bin=False, title='default'):
        po, pa, err, fl1 = self.fit_all_pos(step=step, mode=mode)
        if (mode is 'db_cons') or (mode is 'db_cons_pos'):
            fl, er = self.ampa2fl(pa[0,:], pa[1,:], err_amp=err[0,:], err_a=err[1,:])
        else:
            fl, er = self.ampa2fl(pa[1,:], pa[2,:], err_amp=err[1,:], err_a=err[2,:])
        c = conti.make_prof(self.lam[0], self.lam[-1],bin=step)[1]
        fl *= self.pix_bw/(abs(po[1]-po[0]))
        er *= self.pix_bw/(abs(po[1]-po[0]))
        ew = -fl/c
        er = er/c
        print(fl)
        p2a, p2b, p3, p4 = params[0], params[1], params[2],params[3]
        fl2 = (po>-p2a)*(po<p2b)
        fl3 = er < p3
        where = np.where(fl1*fl2*fl3 == True)
        print(where)
        print(po[where])
        plt.fill_between(po[where], ew[where]-er[where], y2=ew[where]+er[where], color='r', alpha=0.3, step='mid')
        plt.step(po[where], ew[where], where='mid')
        print(np.min(fl[where]), np.max(fl[where]))
        plt.axis([-5.91, 6.08, np.min(ew[where])-0.1*abs(np.min(ew[where])), np.max(ew[where])+0.1*abs(np.max(ew[where]))])
        plt.xlabel('Position relative to photocenter (arcsecond)')
        plt.ylabel('Equivalent width (nm)')
        if lognorm == 'x':
            plt.semilogx()
        if lognorm == 'y':
            plt.semilogy()
        if lognorm == 'xy':
            plt.loglog()
        if title == 'default':
            title = 'Equivalent width  measured by gaussian adjustment of '+str(self.wvl)
        plt.title(title, fontsize=11, fontweight="bold")
        if save_fig:
            plt.savefig(save_nam)
            plt.close()

    def plot_ew(self, conti, step=0.15, save_fig=False, save_nam='./', lognorm=False, bin=False, title='default'):
        po, fl, er = self.make_prof(bin=step)
        c = conti.make_prof(self.lam[0], self.lam[-1], bin=step)[1]#/(conti.pix_bw*(self.lam[-1]-self.lam[0]))
        ew = -fl/c
        er = er/c
        where=np.where(fl>-10)
        plt.fill_between(po[where], ew[where]-er[where], y2=ew[where]+er[where], color='r', alpha=0.3, step='mid')
        plt.step(po[where], ew[where], where='mid')
        print(np.min(fl[where]), np.max(fl[where]))
        plt.axis([-5.91, 6.08, np.min(ew[where])-1*abs(np.min(ew[where])), np.max(ew[where])+1*abs(np.max(ew[where]))])
        plt.xlabel('Position relative to photocenter (arcsecond)')
        plt.ylabel('Equivalent width (nm)')
        if lognorm == 'x':
            plt.semilogx()
        if lognorm == 'y':
            plt.semilogy()
        if lognorm == 'xy':
            plt.loglog()
        if title == 'default':
            title = 'Equivalent width  measured of '+str(self.wvl)
        plt.title(title, fontsize=11, fontweight="bold")
        if save_fig:
            plt.savefig(save_nam)
            plt.close()
