# ComBoNormal: Composite Boxel Normal Identification

Felix Fritzen <fritzen@simtech.uni-stuttgart.de>

### Related publication
S. Keshav, F. Fritzen, M. Kabel: FFT-based Homogenization at Finite Strains using Composite Boxels (ComBo), submitted to Computational Mechanics on April 28, 2022 *(arXiv preprint: http://arxiv.org/abs/2204.13624)*


## Scope
Composite Voxels (cf. Kabel (2015)) have shown to be a meaningful addition to the successful range of FFT-based solvers going back to the seminal work Moulinec (1998). In essence, a high resolution regular grid can benefit from coarse-graining and processing voxels on the interface of adjacent materials via classical laminate theory building on the established Hadamard compatibility and jump conditions.

This software offers a simple yet accurate method for the computation of the coarse-grained structure alongside the normal vectors. HDF5 files are accepted as inputs. A comprehensive API is provided and examples are given within `jupyter` notebooks. The results can easily post-processed in `paraview` and a direct HDF5 export is part of the offered utility. Instructions for the postprocessing are contained in the `jupyter` notebook including a short tutorial video.

![image](https://media.github.tik.uni-stuttgart.de/user/397/files/894cb180-9b4b-11ec-91e5-e718b43d0bf9)

### References

(Moulinec (1998)) H. Moulinec, P. Suquet: A numerical method for computing the overall response of nonlinear composites with complex microstructure, Computer Methods in Applied Mechanics and Engineering 157, p. 69-94, DOI: https://doi.org/10.1016/S0045-7825(97)00218-1.

(Kabel (2015)) M. Kabel, D. Merkert, M. Schneider: Use of composite voxels in FFT-based homogenization, Computer Methods in Applied Mechanics and Engineering 294, p. 168/188, DOI: https://doi.org/10.1016/j.cma.2015.06.003

## License information

The software is released under a 2 clause FreeBSD license, see LICENSE file.
