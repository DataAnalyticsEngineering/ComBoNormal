#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 07:32:30 2022

@author: Felix Fritzen
"""

import numpy as np
import matplotlib.pyplot as plt

def show_microstructure( img, L=[1,1,1], idx=None ):
    """
    Show two slices of the image (with normal being the z-axis).
    If no slice indices are provided, i.e. if idx==None, then
    the first an the center slice will be shown.

    Input images should have value 0 for the matrix phase and value 1 for the
    inclusions.

    :param img: input image, shape=[nz, ny, nx], dtype can, e.g., be float, int, uint8
    :type img: nd-array
    :param L: size of the RVE L=[Lz,Ly,Lx], defaults to [1,1,1]
    :type L: nd-array, list or tuple, optional
    :param idx: indices of the slices; a modulu operation will be applied to
    recover admissible indices; if None --> first clise and the one closest to
    nz/2, defaults to None; datatype must be int-like
    :type idx: nd-array, list or tuple, optional
    :return: None
    :rtype: None

    """
    n = img.shape
    if( idx is None ):
        idx = [0, int( n[0]/2) ] # two slices: bottom and center along z axis
    else:
        assert( len(idx) == 2),'expecting two indices for microstructure slicing'

    fig, ax = plt.subplots(1,2,figsize=(12,6))
    z   = L[0]*(np.linspace(0,1.0,n[0]+1)+0.5/n[0]-0.5)[:-1]
    for (iz, a) in zip(idx,ax.flatten()):
        iz = np.mod( iz, n[0]) # make sure indices are admissible
        img_slab = img[iz,:,:]
        print('image slab at z = %5.2f' % z[iz], '  vol. fraction in slice: %5.2f %%' % (img_slab.mean()*100))
        a.imshow( img_slab[::-1,:], extent=[ -0.5*L[2], 0.5*L[2], -0.5*L[1], 0.5*L[1] ], \
                 interpolation='nearest', cmap=plt.cm.jet, vmin=0, vmax=1)
        a.set_title('slice at z=%5.2f' % ( z[iz] ))
        a.axis()
        a.set_xlabel('$x$ [-]')
        a.set_ylabel('$y$ [-]')
    fig.tight_layout()


    fig, ax = plt.subplots(1,1,figsize=(12,1.5))
    ax.text(0.2,  0.5, 'blue: matrix (0)', fontsize=16)
    ax.text(0.6,  0.5, 'red: inclusion (1)', fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    fig.tight_layout()

# postprocessing for the composite voxels:
def show_slices( vol_frac, combi_img, normals, L=[1,1,1], idx = None ):
    """
    Visualize two plots of two slices of the microstructure:
    plot 1: material index in the composite image with normals (cf. [1] and [2] superimposed)
    plot 2: volume fraction within the composite voxels

    :param vol_frac: volume fraction data for the inclusion phase, shape=[nz, ny, nx], dtype=float
    :type vol_frac: nd-array
    :param combi_img: material indices: 0->matrix, 1->inclusion, 2->composite
    material, shape=[nz, ny, nx], dtype=int or uin8
    :type combi_img: nd-array
    :param normals: normal vectors (new algorithm),
    N[iz,iy,ix,:] --> [normal_x, normal_y, normal_z]
    shape=[nz, ny, nx, 3], dtype=float
    :type normals: nd-array
    :param L: size of the RVE L=[Lz,Ly,Lx], defaults to [1,1,1]
    :type L: nd-array, list or tuple, optional
    :param idx: indices of the slices; a modulu operation will be applied to
    recover admissible indices; if None --> first clise and the one closest to
    nz/2, defaults to None; datatype must be int-like
    :type idx: nd-array, list or tuple, optional
    :return: None
    :rtype: None

    """
    n_combi = vol_frac.shape
    if( idx is None ):
        idx = [0, int( n_combi[0]/2) ] # two slices: bottom and center along z axis
    else:
        assert( len(idx) == 2),'expecting two indices for microstructure slicing'
    fig, ax = plt.subplots(1,2,figsize=(12,6))

    x_c = L[2]*(np.linspace(0,1.0,n_combi[2]+1)+0.5/n_combi[2]-0.5)[:-1]
    y_c = L[1]*(np.linspace(0,1.0,n_combi[1]+1)+0.5/n_combi[1]-0.5)[:-1]
    z_c = L[0]*(np.linspace(0,1.0,n_combi[0]+1)+0.5/n_combi[0]-0.5)[:-1]


    h = np.sqrt( (x_c[1]-x_c[0])**2 + (y_c[1]-y_c[0])**2 ) * 2
    for (iz, a) in zip(idx,ax.flatten()):
        iz = np.mod( iz, n_combi[0]) # make sure indices are admissible
        img_slab = combi_img[iz,:,:].astype(float)
        a.imshow( img_slab[::-1,:], extent=[ -0.5*L[2], 0.5*L[2], -0.5*L[1], 0.5*L[1] ], \
                 interpolation='nearest', cmap=plt.cm.jet, vmin=0, vmax=2)
        iy, ix = np.where( combi_img[iz, :, : ] == 2 )
        X = x_c[ix]
        Y = y_c[iy]
        NX= normals[iz,iy,ix,0] * h
        NY= normals[iz,iy,ix,1] * h
        a.quiver(X,Y,NX,NY, color='magenta', scale=1)
        a.set_title('material index at slice at z=%5.2f' % ( z_c[iz] ))
        a.axis()
        a.set_xlabel('$x$ [-]')
        a.set_ylabel('$y$ [-]')
    fig.tight_layout()

    fig, ax = plt.subplots(1,1,figsize=(12,1.5))
    ax.text(0.05,  0.5, 'blue: matrix (0)', fontsize=16)
    ax.text(0.30,  0.5, 'green: inclusion (1)', fontsize=16)
    ax.text(0.55, 0.5, 'red: composite (2)', fontsize=16)
    ax.text(0.80, 0.5, 'arrows: normals cf. [2]', fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    fig.tight_layout()

    fig, ax = plt.subplots(1,2,figsize=(12,6))

    for (iz, a) in zip(idx,ax.flatten()):
        iz = np.mod( iz, n_combi[0]) # make sure indices are admissible
        img_slab = vol_frac[iz,:,:].astype(float)
        im=a.imshow( img_slab[::-1,:], extent=[ -0.5*L[2], 0.5*L[2], -0.5*L[1], 0.5*L[1] ], \
                 interpolation='nearest', cmap=plt.cm.jet, vmin=0, vmax=1)
        a.set_title('volume fractions at slice at z=%5.2f' % ( z_c[iz] ))
        a.axis()
        a.set_xlabel('$x$ [-]')
        a.set_ylabel('$y$ [-]')
    fig.tight_layout()

    fig, ax = plt.subplots(1,1,figsize=(12,1.5))
    cbar = fig.colorbar(im, ax=ax, cax=ax, orientation='horizontal')
    cbar.set_label('inclusion volume fraction',fontsize=16)
    fig.tight_layout()