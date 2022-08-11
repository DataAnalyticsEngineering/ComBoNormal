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

"""

XDMF H5

by Felix Fritzen, fritzen@simtech.uni-stuttgart.de

Content
=======
This module provides simplistic tools for the visualization of 3D data
within paraview. It is built on HDF5 files and XDMF configuration files
in order to make the data readable in python.

See the demo() function for an example.

"""

import numpy as np
import h5py
from os.path import exists

def write_xdmf(fname, L=[1.,1.,1.], N=[1,1,1], fields=None, labels=None, h5name=None, \
    order_xyz=True, genh5file=False, genxdmffile=False ):

    """
    Generate an XDMF file (to be written, e.g., to an ASCII file) which
    can be used to visualize 3D datasets stored in hdf5 files and
    that are based on regular grids. The file can then, e.g., be accessed
    via Paraview (select "XDMF Reader" from the dialog on open; file
    suffix should be ".xdmf").

    NOTE on ordering: see parameter order_xyz

    !! Within the HDF5 file the order is ALWAYS z-y-x in order to match paraview !!

    :param fname: file name of the HDF5 file containing the data
    :type fname: string
    :param L: Length of the 3D domain, defaults to [1.,1.,1.]
    :type L: ndarray-like, optional
    :param N: number of elements, defaults to [1,1,1]
        (attention: this must match the H5 dataset size)
    :type N: ndarray-like, optional
    :param fields: each item is one ndarray that will be stored within the HDF5
        file (need for the dimensions and to determine dtype), defaults to None
    :type fields: list, optional
    :param labels: each item is the string-type label to one of the datasets from fields, defaults to None
    :type labels: list, optional
    :param h5name: each item is a string denoting the name of the dataset within the HDF5
       file (see parameter fname), defaults to None
    :type h5name: list, optional
    :param order_xyz: order_xyz == True  --> L=[L_x, L_y, L_z], N=[N_x,N_y,N_z] and
                            fields have dimension [N_x,N_y,N_z[, d]]
                            (optional output dimension d>1)
        order_xyz == False --> L=[L_z, L_y, L_x], N=[N_z,N_y,N_x] and
                            fields have dimension [N_z,N_y,N_x[, d]]
                            (optional output dimension d>1)
    :type order_xyz: bool, defaults to True
    :param genh5file: if True, generate the required minimal h5 file
    :type genh5fyle: bool, defaults to False
    :param genxdmffile: if True, the a new file named fname + '.xdmf' will be generated
        (overwritten if existing) containing the generated XDMF XML code
    :type genxdmffile: bool defaults to False
    :return: list of strings (each item corresponding to one line)
    :rtype: list

    **EXAMPLE**

    write_xdmf( 'example.h5', L=[1.,1.,2.], N=[8,8,16], \
               fields=[material, u], \
               labels=['MATERIAL_NUMBER','SOLUTION'], \
               h5name=['mesh/material','solution/u'])

    Expects the existing of an HDF5 file containing to datasets:
        /mesh/material    --> 16*8*8 ndarray (since z,y,x ordering!)
        /solution/u       --> 16*8*8 ndarray (since z,y,x ordering!)
    The ndarrays material and u must be provided (shape=[8,8,16] and proper
                                                  dtype are expected)
    In the visualizer (e.g. Paraview) the two field will have names
    'MATERIAL_NUMBER' and 'SOLUTION', respectively.

    see also demo() below

    ATTENTION: Avoid use of whitespaces (can/will cause trouble).
    """

    N = np.array(N)
    L = np.array(L)
    assert( N.min() > 0 ), 'expecting positive range for all entries of N - parameter order might be confused?'
    assert( N.size == 3 ), 'N is expected to be a three-dimenionsal array-like input denoting the grid size - parameter order might be confused?'
    assert( L.min() > 0 ), 'expecting positive range for all entries of L - parameter order might be confused?'
    assert( L.size == 3 ), 'L is expected to be a three-dimenionsal array-like input denoting the domain edge lengths - parameter order might be confused?'

    DIM = N.copy()
    if( order_xyz ):
        # invert order of DIM (must be ZYX) and L
        DIM = DIM [::-1]
        L = L[::-1]

    # print('L=', L, '  DIM=', DIM)

    s = []
    s.append('<Xdmf Version="2.2">')
    s.append('  <Domain>')
    s.append('    <Grid Collection="0" GridType="Uniform">')
    s.append('      <Topology Dimensions="%d %d %d" TopologyType="3DCORECTMesh" />' % (DIM[0]+1,DIM[1]+1,DIM[2]+1) )
    s.append('      <Geometry GeometryType="ORIGIN_DXDYDZ">')
    s.append('        <DataItem Dimensions="3" Format="XML" Name="Origin" NumberType="Float" Precision="4">%f %f %f</DataItem>' % (-L[2]/2., -L[1]/2., -L[0]/2.))
    s.append('        <DataItem Dimensions="3" Format="XML" Name="Spacing" NumberType="Float" Precision="4">%f %f %f</DataItem>' % (L[0]/DIM[0], L[1]/DIM[1], L[2]/DIM[2]))
    s.append('      </Geometry>')
    for [f, h, lab] in zip(fields, h5name, labels):
        assert( np.all( N == np.array(f.shape[:3])) ), 'array dimension and given grid size do not match (N ~= shape[:3] of data)'
        if( f.ndim==3 ):
            s.append('      <Attribute AttributeType="Scalar" Center="Cell" Name="%s">' % (lab))
            s.append('        <DataItem')
            s.append('                Dimensions="%d %d %d"' % (DIM[0], DIM[1], DIM[2]))
        if( f.ndim==4 ):
            s.append('      <Attribute AttributeType="Vector" Center="Cell" Name="%s">' % (lab))
            s.append('        <DataItem')
            s.append('                Dimensions="%d %d %d %d"' % (DIM[0], DIM[1], DIM[2], f.shape[3]))
        s.append('                Format="HDF"')
        if( f.dtype==np.uint8 or f.dtype==bool):
            s.append('                NumberType="UChar">')
        elif( f.dtype==int):
            s.append('                NumberType="Int"')
            s.append('                Precision="4">')
        elif( f.dtype==float):
            s.append('                NumberType="Float"')
            s.append('                Precision="8">')
        else:
            print('dtype:', f.dtype, ', ', h, ', ', lab)
        s.append('                %s:/%s' % (fname, h))
        s.append('        </DataItem>')
        s.append('      </Attribute>')
    s.append('    </Grid>')
    s.append('  </Domain>')
    s.append('</Xdmf>')

    if( genh5file ):
        hdf = h5py.File( fname, 'w')
        compression = { 'compression' : 'gzip' , 'compression_opts' : 9}
        for [f, h ] in zip(fields, h5name):
            if( order_xyz ):
                if(f.ndim==4):
                    hdf.create_dataset(h, data=f.transpose((2,1,0,3)), **compression)
                else:
                    hdf.create_dataset(h, data=f.transpose(), **compression)
            else:
                hdf.create_dataset(h, data=f, **compression)
        hdf.close()

    if( genxdmffile ):
        fn = fname + '.xdmf'
        str2file( fn, s, overwrite=True)
    return s

def str2file(fname, s, overwrite=True):
    """
    :param fname: filename of the text file to be generated
    :type fname: string
    :param s: list of strings (each list entry will by treated as one line)
    :type s: list
    :param overwrite: if True, the file named fname will be overwritten
    :type overwrite: bool, defaults to True
    """
    if( not overwrite ):
        assert( not exists(fname) ),'str2file received "no overwrite flag", but file exists - aborting'
    f = open(fname,'w')
    for x in s:
        print(x,file=f)
    f.close()


def demo():
    """ illustrates how to use write_xdmf"""

    Limg = np.array([1.,1.,2.])
    material = np.random.randint(low=0, high=3, size=[8,8,16], dtype=np.uint8)
    u =np.random.normal(0., 1., size=[8,8,16])


    # More detailled version:
    # can be useful if h5 files are appended etc.
    #hdf=h5py.File('example.h5','w')
    #compression = { 'compression' : 'gzip' , 'compression_opts' : 9}
    #hdf.create_dataset('mesh/material', data=material.transpose(), **compression)
    #hdf.create_dataset('solution/u', data=u.transpose(), **compression )
    #hdf.close()

    #s = write_xdmf('example.h5', L=Limg, N=material.shape, \
                        #h5name=[ 'mesh/material', 'solution/u'], \
                        #fields=[material, u], \
                        #labels=['MATERIAL_NUMBER','SOLUTION'])
    #str2file('example.h5.xdmf', s, overwrite=True)

    #short version:
    s = write_xdmf('example.h5', L=Limg, N=material.shape, \
              h5name=[ 'mesh/material', 'solution/u'], \
              fields=[material, u], \
              labels=['MATERIAL_NUMBER','SOLUTION'], \
              genh5file=True, genxdmffile=True)
    print('example data written to "example.h5" and related XDMF file for paraview is stored as "example.h5.xdmf"')


