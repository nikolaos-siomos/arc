#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:49:58 2023

@author: nikos
"""
import numpy as np
from numpy.random import multivariate_normal as normal_2D
import xarray as xr

def angle_of_incidence(theta_x, theta_y, height, epsilon, delta, n_rays):
    # omega: angle of incidence at the telescope apperture (parallel beam) in mrad
    omega = xr.DataArray(dims = ('theta_x', 'theta_y', 'height', 'rays'),
                         coords = [theta_x, theta_y, height, np.arange(n_rays)])
    
    for i in range(theta_x.size):
        for j in range(theta_y.size):
            mean = [theta_x[i],theta_y[j]]
            
            cov = [[np.square(epsilon / 2.) ,0.],[0., np.square(epsilon / 2.)]]
            
            gaussian_beam = normal_2D(mean, cov, n_rays)
            
            theta_x_n = gaussian_beam[:,0]
            
            theta_y_n = gaussian_beam[:,1]
            
            for k in range(height.size):
                omega[i,j,k,:] = \
                    1e3 * np.arctan(np.sqrt(np.tan(1e-3 * theta_x_n)**2 + \
                        (1e-6 * delta / height[k] - np.tan(1e-3 * theta_y_n))**2))
                omega[i,j,k,0] = \
                    1e3 * np.arctan(np.sqrt(np.tan(1e-3 * theta_x[i])**2 + \
                        (1e-6 * delta / height[k] - np.tan(1e-3 * theta_y[j]))**2))

    return(omega)

def angle_of_incidence_IF(omega, f_tel, f_col):

    omega_if = 180. * 1e-3 * omega * (f_tel / f_col) / np.pi
    
    return(omega_if)

def change_tilt(omega, theta_t, theta_z, D, M):
    
    height = omega.height.values 

    phi = 1e-6 * D / height
    
    # omega_t = xr.DataArray(dims = omega.dims, coords = omega.coords)
    # omega_i = omega.copy() * np.pi / 180.
    
    omega_t_sim = np.nan * omega.copy()
    
    omega_z = 180. * np.arctan(phi - np.tan(1e-3 * theta_z)) * M / np.pi
    omega_t = 180. * np.arctan(phi - np.tan(1e-3 * theta_t)) * M / np.pi
    
    for i in range(height.size):
        omega_t_sim[i,:] = omega[i,:] - omega_z[i] + omega_t[i]
    
    # for i in range(height.size):
    #     theta = np.arctan(phi[i] - np.tan(omega_i.values[i,:] / M))
    #     omega_t[i,:] = np.arctan(phi[i] - np.tan(theta - 1e-3 * theta_z + 1e-3 * theta_t)) * M 
    
    # omega_t = 180. * omega_t / np.pi
    
    return(omega_t_sim)
