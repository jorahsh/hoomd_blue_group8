// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "DyBondUpdater.cuh"

/*! \file DyBondUpdater.cu
    \brief CUDA kernels for DyBondUpdater
*/

// First, the kernel code for zeroing the velocities on the GPU
//! Kernel that zeroes velocities on the GPU
/*! \param d_vel Velocity-mass array from the ParticleData
    \param N Number of particles

    This kernel executes one thread per particle and zeros the velocity of each. It can be run with any 1D block size
    as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__
void gpu_zero_velocities_kernel(volatile bool *found,
                                Scalar4 *d_pos,
                                unsigned int N, 
                                const group_storage<2> *blist,
                                const unsigned int pitch,
                                const unsigned int *n_bonds_list,
                                const unsigned int n_bond_type,
                                const unsigned int *d_n_neigh,
                                const unsigned int *d_nlist,
                                const unsigned int *d_head_list,
                                const unsigned int size_nlist)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("in cuda kernel. blockIdx.x:%d,blockDim.x:%d,threadIdx.x:%d\n",blockIdx.x,blockDim.x,threadIdx.x);
    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    if (idx < N)
        {
        // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
        Scalar4 idx_postype = d_pos[idx];  // we can be either a, b, or c in the a-b-c-d quartet
        unsigned int bond_from_type = __scalar_as_int(idx_postype.w);
        if (bond_from_type != 2)
	    {
            int bond_to_type = (bond_from_type==0)?1:0;
            // load in the length of the list
    	    unsigned int n_neigh = d_n_neigh[idx];
    	    const unsigned int head_idx = d_head_list[idx];
            //printf("particle id:%d, n_neigh:%d, head_idx:%d\n",idx,n_neigh,head_idx);

            unsigned int next_neigh(0);
            // loop over neighbors
            for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
                 {
                 next_neigh = d_nlist[head_idx + neigh_idx + 1];
                 Scalar4 idx_postype = d_pos[next_neigh];  // we can be either a, b, or c in the a-b-c-d quartet
                 unsigned int ptype = __scalar_as_int(idx_postype.w);
                 if (ptype==bond_to_type)
                     {
                     printf("Bond from %d(%d) to %d(%d)\n",idx,bond_from_type,next_neigh,bond_to_type);
                     // MEM TRANSFER: 8 bytes
                     group_storage<2> cur_bond = blist[0];
                     printf("cur_bond %d,%d\n",cur_bond.idx[0],cur_bond.idx[1]);
                     }
                 }
            }        
        }
        
    }

/*! \param d_vel Velocity-mass array from the ParticleData
    \param N Number of particles
    This is just a driver for gpu_zero_velocities_kernel(), see it for the details
*/
cudaError_t gpu_zero_velocities(Scalar4 *d_pos, 
				unsigned int N,
				const group_storage<2> *blist,
                const unsigned int pitch,
                const unsigned int *n_bonds_list,
                const unsigned int n_bond_type,
                const unsigned int *d_n_neigh,
                const unsigned int *d_nlist,
                const unsigned int *d_head_list,
                const unsigned int size_nlist)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (int)ceil((double)N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    volatile bool found = false;
    bool *foundptr = (bool*) &found;
    // run the kernel
    printf("Going to call the kernel\n");
    gpu_zero_velocities_kernel<<< grid, threads >>>(foundptr,
                                                    d_pos,
                                                    N,
                                                    blist,
                                                    pitch,
                                                    n_bonds_list,
                                                    n_bond_type,
						    d_n_neigh,
						    d_nlist,
						    d_head_list,
						    size_nlist);

    // this method always succeds. If you had a cuda* call in this driver, you could return its error code if not
    // cudaSuccess
    return cudaSuccess;
    }
