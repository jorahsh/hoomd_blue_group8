// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _DYBOND_UPDATER_CUH_
#define _DYBOND_UPDATER_CUH_

// need to include the particle data definition
#include <hoomd/ParticleData.cuh>
#include "hoomd/BondedGroupData.cuh"

/*! \file DyBondUpdater.cuh
    \brief Declaration of CUDA kernels for DyBondUpdater
*/

// A C API call to run a CUDA kernel is needed for DyBondUpdaterGPU to call
//! Zeros velocities on the GPU
extern "C" cudaError_t gpu_zero_velocities(Scalar4 *d_vel, 
 					   unsigned int N,
					   const group_storage<2> *blist,
                                           const unsigned int pitch,
                                           const unsigned int *n_bonds_list,
                                           const unsigned int n_bond_type,
                                           const unsigned int *d_n_neigh,
                                           const unsigned int *d_nlist,
                                           const unsigned int *d_head_list,
                                           const unsigned int size_nlist);

#endif // _DYBOND_UPDATER_CUH_
