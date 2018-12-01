// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _DYBOND_UPDATER_H_
#define _DYBOND_UPDATER_H_

/*! \file DyBondUpdater.h
*/
#include "hoomd/md/NeighborList.h"
#include "hoomd/ComputeThermo.h"
#include "hoomd/ParticleGroup.h"
#include <hoomd/Updater.h>
#include <map>
#include <random>
#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

//! An updater for dynamically creating bonds between pair of particle types with a given probability and frequency.
class DyBondUpdater : public Updater
    {
    public:
        //! Constructor
        DyBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<NeighborList> nlist,
                      std::shared_ptr<ComputeThermo> thermo);
        //! Take one timestep forward
        virtual void update(unsigned int timestep);
        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities(void);
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);
        //! set the parameters that are not set in the constructor
        void set_params(std::string bond_type,
                        std::string A,
                        unsigned int A_fun_groups,
                        std::string B,
                        unsigned int B_fun_groups,
                        Scalar rcut,
                        Scalar Ea,
                        Scalar alpha,
                        Scalar percent_bonds_per_step,
                        Scalar callback_at_percent,
                        pybind11::object callback,
                        bool exclude_from_nlist,
                        bool enable_enthalpy_change,
                        Scalar deltaT);
        template <typename T>
        std::vector<size_t> sort_indexes(const std::vector<T> &v);
        std::map<unsigned int, pybind11::object> m_callback;

    protected:
        std::mt19937 m_rnd;
        unsigned int m_seed;                                //!< Random seed to use
        unsigned int bond_rank(unsigned int p_idx);
        Scalar get_bond_percent(unsigned int bond_type);
        void init_dictionaries_from_system(unsigned int bond_type);
        bool is_bondable_particle(unsigned int b_type, unsigned int p_type);
        Scalar addHeat(unsigned int timestep,
                    unsigned int num_bonds_created,
                    Scalar current_temperature);
        std::shared_ptr<BondData> m_bond_data;    //!< Bond data to use in computing bonds
        std::map<unsigned int,std::pair<unsigned int, unsigned int> > m_bond_type; //!< Dictionary of bond type and the participating particle types
        std::map<unsigned int, unsigned int> m_functional_groups; //!< Dictionary of particle type and number of functional groups
        std::map<unsigned int, unsigned int> m_possible_bonds; //!< Key is bond type and value is number of possible bonds
        std::map<unsigned int, unsigned int> m_bonds_made; //!< Key is bond type and value is number of bonds made by the system.
        std::vector<std::string> m_loggable_quantities;
        std::map<unsigned int, Scalar> m_stop_after_percent;
        std::map<unsigned int,bool> m_stop_bonding;
        std::map<unsigned int, Scalar> m_max_bonds_per_attempt;
        std::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        const std::shared_ptr<ComputeThermo> m_thermo; //!< compute for thermodynamic quantities
        Scalar m_Ea;//activation_energy
        Scalar m_alpha;//secondary_bond_weight
        Scalar m_rcut;//cut off distance for bonding
        //unsigned int m_possible_bonds = 0;
        std::map<unsigned int, unsigned int> m_rank_dict; //!< stores bond rank for particle ids to speed up lookup
        std::map<unsigned int, Scalar> m_summed_bps; //!< sum of bonds per step for calculated bps <1
        bool m_exclude_from_nlist=false;
        bool m_enable_enthalpy = true;
        Scalar m_deltaT;
        unsigned int m_num_bonds_per_step=0;//used for logging. Just stores the number of bonds created in the last iteration
    };

//! Export the DyBondUpdater class to python
void export_DyBondUpdater(pybind11::module& m);

// Third, this class offers a GPU accelerated method in order to demonstrate how to include CUDA code in pluins
// we need to declare a separate class for that (but only if ENABLE_CUDA is set)

#ifdef ENABLE_CUDA

//! A GPU accelerated version of the DyBondUpdater.
class DyBondUpdaterGPU : public DyBondUpdater
    {
    public:
        //! Constructor
        DyBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<NeighborList> nlist,
                         std::shared_ptr<ComputeThermo> thermo);
        //! Take one timestep forward
        virtual void update(unsigned int timestep);
    };

//! Export the DyBondUpdaterGPU class to python
void export_DyBondUpdaterGPU(pybind11::module& m);

#endif // ENABLE_CUDA

#endif // _DYBOND_UPDATER_H_
