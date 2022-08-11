#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Felix Fritzen (fritzen@simtech.uni-stuttgart.de)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

-----------------------------

This software is related to the following research article:

S. Keshav, F. Fritzen, M. Kabel: FFT-based Homogenization at
Finite Strains using Composite Boxels (ComBo),
submitted to Computational Mechanics on April 28, 2022

It can be accessed via https://github.tik.uni-stuttgart.de/DAE/ComBoNormal

-----------------------------
"""

import numpy as np
import h5py
import os
from numpy.fft import fftn,ifftn,rfftn,irfftn
from xdmfh5 import write_xdmf

def supervoxel_normal(img, lap_img, l, vol_frac=None):
    """Get the normal vector from a supervoxel (given by img) and using the
    Laplacian on the image (lap_img). The Laplacian can be larger in size in order
    to get more accurate surface information (only for smooth surfaces),
    particularly if the supervoxels are small (resolution <= 4 for any axis).
    Then a least squares problem is solved to find the optimal normal vector.

    The normal is returned in (x,y,z) convention. The established normal cf.
    the approach of Kabel et al. (2015) is returned (ATTENTION: characteristic
    length is not adjusted for that).

    The normal points from the inclusion phase (1) into the matrix phase (0).

    :param img:     input pixels of the supervoxel under consideration (binary)
    :type img:      ndarray
    :param lap_img: like img, but the Laplacian of the img and (optional) of a different size
    :type lap_img:  ndarray
    :param l:       edge length of the fine scale voxels (lz, ly, lx), dtype=float
    :type l:        ndarray
    :param vol_frac: volume fraction of phase 1 (computed if None),
                    range: 0.0-1.0, defaults to None.
    :type vol_frac: float, optional
    """
    inline_norm = lambda x: np.sqrt( x[0]*x[0] +x[1]*x[1]+x[2]*x[2])

    assert(img.ndim == 3),'error: expecting 3D ndarray (0--> phase 0; else-->phase 1)'
    N       = np.array(img.shape)      # size of the supervoxel
    l_combi = l*np.array(img.shape)    # edge lengths of the supervoxel
    # assemble and solve the least squares problem
    w       = np.abs(lap_img)          # weights
    iface   = np.where(w > 1e-5)       # find the interface voxels
    w       = w[iface[0],iface[1],iface[2]].flatten()
    w       = w/w.sum()                # renormalize
    # coordinates of the 'on interface' voxels
    XX      = l[:,None]*np.array(iface)
    Xbar    = (XX*w).sum(axis=1)
    XX      = XX - Xbar[:,None]
    Mmod    = XX@(XX*w).T
    eigval, evec   = np.linalg.eigh(Mmod)
    EV       = evec[:,np.argmin(eigval)]
    # reorder: normal will have ordering (x, y, z) after the following line
    normal_xyz  = EV[::-1] / np.sqrt(EV[0]*EV[0]+EV[1]*EV[1]+EV[2]*EV[2])

    # concentration of phase '1' and its barycenter
    p_sum   = img.sum(axis=(1,2)) # x-y-slice sum
    n_red   = N[0]*N[1]*N[2]      # number of voxels in the supervoxel
    if( vol_frac is None ):
        c1  = p_sum.sum()/n_red
    else:
        c1  = vol_frac

    # a fast implementation that avoids redundant sub-summation
    # x1_c/y1_c/z1_c: barycenter of phase 1
    x       = []
    for i in range(3):
        x.append( np.linspace(-.5, .5, N[i]+1 )[:-1] + 0.5/ N[i]  )

    z1_c    = (p_sum*x[0]).sum()/(n_red*c1) * l_combi[0]

    p_sum   = img.sum(axis=0) # z-sum

    y1_c    = (p_sum.sum(axis=1)*x[1]).sum()/(n_red*c1) * l_combi[1] # x-sum
    x1_c    = (p_sum.sum(axis=0)*x[2]).sum()/(n_red*c1) * l_combi[2] # y-sum

    # flip direction if the normal points into phase 1:
    if( x1_c*normal_xyz[0] + y1_c*normal_xyz[1] + z1_c*normal_xyz[2] > 0. ):
        normal_xyz  *= -1.

    normal_classic = np.array([x1_c, y1_c, z1_c])
    normal_classic = -normal_classic / inline_norm(normal_classic)
    return normal_xyz, normal_classic


#----------------------------------------------------------------------
def supervoxelize_image( img, L = [1.,1.,1.], \
                        voxel_size=[8,8,8], \
                        hdfname=None, groupname=None, overwrite=False, \
                        gzip=True, quiet=True ):
    """
    This function accepts a binary image (img). It downsamples this image
    by using composite voxels/boxels [1, 2]. First, the obvious downsampling
    is performed for the coarse voxels/boxels with unique material assignment.
    Then the interface of the material in the finescale image is detected
    by using a discrete Laplace stencil. The normal vectors inside of the
    composite voxels/boxels are gathered from an optimization problem cf. [2].

    By definition the normals are facing *OUT OF PHASE 1*.

    The results are stored in HDF5 files on request which enable easy
    visualization using, e.g., Paraview in combination with automatically
    generated XDMF files.

    All inputs and outputs have a data format that is built on z,y,x indexing,
    i.e. img[iz,iy,ix]. Internally, vectors are also expected in z,y,x
    ordering. This has implications on the edge length of the cuboid on input
    etc.

    [1] M. Kabel, D. Merkert, M. Schneider: Use of composite voxels
        in FFT-based homogenization, Computer Methods in Applied Mechanics
        and Engineering, Volume 294, p. 168-188, 2015.
        DOI: https://doi.org/10.1016/j.cma.2015.06.003
    [2] S. Keshav, F. Fritzen, M. Kabel: , Computational Mechanics,
        submitted (2022), preprint: arxiv, XXX-XXX


    :param img:         input image, shape=[ nz, ny, nx ], dtype=uint8
    :type img:          nd-array
    :param L:           edge length of the cube L=[ Lz, Ly, Lx ],
                        shape = [ 3 ], dtype = float, defaults to [ 1., 1., 1. ]
    :type L:            list or nd-array, optional
    :param voxel_size:  size of the super voxel [sz, sy, sx], defaults to [8,8,8]
    :type voxel_size:   list or nd-array, optional
    :param hdfname:     name of the HDF5 file, defaults to None
    :type hdfname:      string, optional
    :param groupname:   name of the group to store the raw image and the
                        supervoxel data (if None: '/'), defaults to None
    :type groupname:    string, optional
    :param overwrite:   overwrite HDF5 file, if file exists, defaults to False
    :type overwrite:    bool, optional
    :param gzip:        enable gzip compression within the HDF5 file, defaults to True
    :type gzip:         bool, optional
    :param quiet:       suppress debug information, defaults to True
    :type quiet:        bool, optional
    :return:            local volume fractions, normal vectors,
                        normal vectors (as in previous works),
                        composite material indices (0: matrix, 1: inclusion,
                        2: composite voxel)
                        ATTENTION: normal vectors (i.e. x-y-z notation)
                        normal = [normalx, normaly, normalz] = NORMAL[iz,iy,ix,:
                        shapes: [nz,ny,nx], [nz,ny,nx,3], [nz,ny,nx,3], [nz,ny,nx]
                        dtype: float, float, float, numpy.uint8
    :rtype:             nd-array, nd-array, nd-array, nd-array

    """
    # perform some consistency checks
    if( hdfname is not None ):
        assert( hdfname.endswith('.h5')),'HDF5 filename should carry suffix ".h5"; please adjust filename to guarantee full operability - thanks!'
    if( not quiet ):
        print('supervoxel_img - input image size: ', img.shape, ', supervoxel size: ', voxel_size)

    voxel_size  = np.array(voxel_size, dtype=int)
    s_img       = np.array(img.shape,dtype=int)

    assert(  (np.mod(s_img, voxel_size) == 0).all()) , \
             "inconsistent mesh resolution/supervoxel size - have you checked that the image dimensions cleanly integer-divide by the supervoxel-size?"

    #----------------------------------------------------------------------
    # compute the resolution of the downscaled image
    s_combi     = (s_img/voxel_size).astype(int)

    #----------------------------------------------------------------------
    combi_img       = np.zeros(s_combi, dtype=float)
    img_min         = img.min()
    img_max         = img.max()
    #----------------------------------------------------------------------
    # some debug output
    if( not quiet ):
        print('supervoxelize img -- overview')
        print('coarse img size    ', s_combi)
        print('input image size   ', s_img )
        print('supervoxel size    ', voxel_size)
        print('combi image size   ', s_combi )
        print('min. phase number  ', img_min, '  (input image)')
        print('max. phase number  ', img_max, '  (input image)')
    assert( (img_min == 0) and (img_max == 1) ), \
        'input image is expected to have phase indices 0 and 1 only - did you provide the binarized image?'
    #----------------------------------------------------------------------
    # binarize image
    img = ( img > 0 )

    # basic downscaling: compute effective volume fraction
    for ix in range(voxel_size[2]):
        for iy in range(voxel_size[1]):
            for iz in range(voxel_size[0]):
                combi_img += img[iz::voxel_size[0],iy::voxel_size[1],ix::voxel_size[2]]
    # divide by number of fine scale voxels per supervoxel
    s = voxel_size.prod()
    combi_img /= float(s)

    #----------------------------------------------------------------------
    # identify all composite voxels and allocate memory for
    # - normal vectors
    # - combo material label (0: background, 1: foregroung, 2: composite, 3: special)
    # - length: characteristic length
    # - area: area of facet intersection
    combo_id    = np.where( (combi_img >= 1e-6) * (combi_img <= (1.0 - 1e-6) ) )
    if( not quiet ):
        print('found ', len(combo_id[0]), ' composite voxels (out of ', combi_img.size, ' combi voxels)')
    NORMAL      = np.zeros( combi_img.shape + (3,), dtype=float )
    NORMAL_classic= np.zeros( combi_img.shape + (3,), dtype=float )
    combi_mat   = np.zeros( combi_img.shape, dtype=np.uint8 )
    #----------------------------------------------------------------------
    # step 1: apply periodic Laplace stencil to the input image
    img_stencil = np.zeros(img.shape)
    L           = np.array(L, dtype=float)
    # dimensions of the finescale voxels
    lz      = L[0]/s_img[0]
    ly      = L[1]/s_img[1]
    lx      = L[2]/s_img[2]
    l_vx    = np.array([lz,ly,lx])

    # setup the stencil, see reference [2]
    # this stencil accounts for heterogeneous grid size along x, y, z
    f   = 3./(lx*ly/lz+lz*ly/lx+lx*lz/ly)
    img_stencil[0,0,0]      += 2.*f*(lx*ly/lz + lx*lz/ly + ly*lz/lx)
    img_stencil[(1,-1),0,0] += - f*lx*ly /lz
    img_stencil[0,(1,-1),0] += - f*lx*lz /ly
    img_stencil[0,0,(1,-1)] += - f*ly*lz /lx

    #----------------------------------------------------------------------
    # In case of an odd number of voxels the stencil can be computed
    # from a real-valued FFT (advantage: less memory effort, ...)
    even = (s_img[2] % 2 == 1)
    if( even ):
        iface= np.abs(ifftn( fftn(img.astype(float))*fftn(img_stencil) ))
    else:
        iface= np.abs(irfftn( rfftn(img.astype(float))*rfftn(img_stencil) ))

    #----------------------------------------------------------------------
    # label all voxels of the foreground (all pixels have same color)
    combi_mat[ combi_img> (1.0-0.5/s) ] = 1
    # voxels with intermediate volume fraction --> composite boxels
    combi_mat[ combo_id[0], combo_id[1], combo_id[2] ] = 2

    # loop over the composite boxels
    for iz,iy,ix in zip( combo_id[0], combo_id[1], combo_id[2]):
        # sx/sy/sz: start of voxel
        sx = ix*voxel_size[2]
        sy = iy*voxel_size[1]
        sz = iz*voxel_size[0]
        supervoxel          = img[ sz:(sz+voxel_size[0]), sy:(sy+voxel_size[1]), sx:(sx+voxel_size[2])]
        supervoxel_iface    = iface[ sz:(sz+voxel_size[0]), sy:(sy+voxel_size[1]), sx:(sx+voxel_size[2])]
        N, N_classic        =  supervoxel_normal( supervoxel, lap_img = supervoxel_iface, l = l_vx, vol_frac = combi_img[iz,iy,ix] )
        NORMAL[iz,iy,ix,:]  = N
        NORMAL_classic[iz,iy,ix,:]  = N_classic

    if( hdfname is not None):
        if( not quiet ):
            print('dumping data to file "%s" (use Paraview to open the two XDMF files using the XDMFReader)' % hdfname)
        if( not overwrite ):
            assert( not os.path.exists(hdfname) ), 'the hdf5 file intended for output exists -- aborting to avoid hazardous data deletion (try: overwrite=True option if needed)'
        if( os.path.exists(hdfname)):
            if( not quiet ):
                print('erasing "%s" for subsequent overwrite' % hdfname)
            if( overwrite ):
                os.remove(hdfname)
                hdf=h5py.File(hdfname,'w')
            else:
                hdf=h5py.File(hdfname,'a')
        else:
            hdf=h5py.File(hdfname,'w')
        compression = { }
        if( gzip ):
            compression = { 'compression' : 'gzip' , 'compression_opts' : 9}

        if( groupname is not None ):
            grp = hdf.require_group( groupname )
        else:
            grp = hdf
        grp.create_dataset('raw/img',           data=img.astype(np.uint8), **compression)
        grp.create_dataset('raw/iface',         data=(iface.astype(float)>0.00001), **compression )
        grp.create_dataset('combi/vol_frac',    data=combi_img.astype(float), **compression)
        grp.create_dataset('combi/normal',      data=NORMAL, **compression)
        grp.create_dataset('combi/normal_classic',data=NORMAL_classic, **compression)
        grp.create_dataset('combi/combimat',    data=combi_mat, **compression)
        hdf.close()

        if( groupname is None):
            groupname = ''
        s = write_xdmf(hdfname, L=L, N=s_img, \
                           h5name=[ '%s/raw/img' % groupname, '%s/raw/iface' % groupname], \
                           fields=[img, iface], \
                           labels=['MATERIAL','IFACE'], \
                           order_xyz = False)
        xdmfname= hdfname[:-3] + '_raw.h5.xdmf'
        f = open(xdmfname,'w')
        for x in s:
            print(x,file=f)
        f.close()
        s = write_xdmf(hdfname, L=L, N=s_combi, \
                        h5name=[ '%s/combi/vol_frac' % groupname, '%s/combi/normal' % groupname, '%s/combi/normal_classic' % groupname, '%s/combi/combimat'  % groupname ], \
                        fields=[combi_img, NORMAL, NORMAL_classic, combi_mat ],
                        labels=['VOL_FRAC','NORMAL','NORMAL_CLASSIC','COMBIMAT' ],
                        order_xyz=False)
        xdmfname= hdfname[:-3] + '_combi.h5.xdmf'
        f = open(xdmfname,'w')
        for x in s:
            print(x,file=f)
        f.close()

    return combi_img, NORMAL, NORMAL_classic, combi_mat
