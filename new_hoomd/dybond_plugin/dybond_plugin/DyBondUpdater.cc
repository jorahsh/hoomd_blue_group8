// Copyright (c) 2017 Stephen Thomas
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
#define USE_DIST_SORTED_NLIST
#define USE_MERSENNE_TWISTER_ENGINE
#include "DyBondUpdater.h"
#ifdef ENABLE_CUDA
#include "DyBondUpdater.cuh"
#endif
#include "hoomd/Saru.h"
#include <math.h>

// windows defines a macro min and max
#undef min
#undef max

using namespace hoomd;
/*! \file DyBondUpdater.cc
    \Dynamic Bond Updater creates bonds between two types of particles at a specified frequency and with a 
     probability defined by the boltzmann factor.
*/


/*! \param sysdef: System to dynamically create bonds
    \param nlist : neighbour list to use for finding particles to bond to
    \param thermo: ComputeThermo object to get temperature of the system
    \param group : The group of particles this updater method is to work on
*/
DyBondUpdater::DyBondUpdater(std::shared_ptr<SystemDefinition> sysdef, 
                             std::shared_ptr<NeighborList> nlist,
                             std::shared_ptr<ComputeThermo> thermo)
        : Updater(sysdef), m_nlist(nlist), m_thermo(thermo)
    {
    // access the bond data for later use
    m_bond_data = m_sysdef->getBondData();
    assert(m_nlist);
    // start the random number generator
    m_seed = 1234;
    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;
    m_rnd.seed(std::mt19937::result_type(m_seed));
    }

/*! Defines a bond type between two particle types that should be creater by the updater
   \params bond_type   : name of the bond type e.g. 'A-B', 'A-A' or 'polymer'
           A     : name of the first particle type participating in the bond e.g. 'A'
           A_fun_groups: number of functional groups in the participating particle types. 
           B     : name of the second particle type participating in the bond e.g. 'C'
           B_fun_groups: number of functional groups in the participating particle types. 
           Ea          : activation energy (in energy units)
           alpha       : secondary bond weight. This value is used to multiply the activation energy for secondary
           bonds. Here secondary bond refers to the situation where either of the participating particles are making a
           second bond.
           percent_bonds_per_step: The percentage of the number of possible bonds that will be attempted per bond
           step. The resultant number will be rounded to the nearest integer to calculate the number of bonds. If the
           calculated number is less than one, it will be forced to one. This parameter makes the reaction kinetics
           independent of the system size. The default value of 0.0025 is equivalent to 1 bond in a system where 40000
           bonds of this type are possible.
           stop_after_percent: After the system has made this much percentage bonds possible, stop making anymore
           bonds. The default value is 100.
           callback: A python callback object which will be called when the "stop_after_percent" is reached.
           exclude_from_nlist: Specifies whether the newly bonded particles should be excluded from the neighbour
           list. For DPD, this is typically "False" because we want the bonded particles to feel the dissipative and
           drag forces which are calculated pairwise. This value may be set to "True" while using a Langevin integrator
           where these dissipative and drag forces are not pairwise for example.
*/
void DyBondUpdater::set_params(std::string bond_type,
                               std::string A,
                               unsigned int A_fun_groups,
                               std::string B,
                               unsigned int B_fun_groups,
                               Scalar rcut,
                               Scalar Ea,
                               Scalar alpha,
                               Scalar percent_bonds_per_step,
                               Scalar stop_after_percent,
                               pybind11::object callback,
                               bool exclude_from_nlist,
                               bool enable_enthalpy_change,
                               Scalar deltaT)
    {
    if (m_bond_type.size() < 1)
        {
        assert(A_fun_groups > 0);
        assert(B_fun_groups > 0);
        assert(rcut >= 0);
        assert(Ea >= 0);
        assert(alpha >= 0);
        assert(percent_bonds_per_step > 0 && percent_bonds_per_step <= 100);
        unsigned int b_type = m_bond_data->getTypeByName(bond_type);
        unsigned int A_type = m_pdata->getTypeByName(A);
        unsigned int B_type = m_pdata->getTypeByName(B);
        m_Ea = Ea;
        m_alpha = alpha;
        m_rcut = rcut;
        m_exclude_from_nlist = exclude_from_nlist;
        m_enable_enthalpy = enable_enthalpy_change;
        m_deltaT = deltaT;
        if (m_stop_bonding.find(b_type) == m_stop_bonding.end())
            m_stop_bonding[b_type] = false;
        if (m_stop_after_percent.find(b_type) == m_stop_after_percent.end())
            m_stop_after_percent[b_type] = stop_after_percent;
            //printf("stop_after_percent has value:%f %f\n",stop_after_percent,m_stop_after_percent[b_type]);
        if (m_callback.find(b_type) == m_callback.end())
            m_callback[b_type] = callback;
        if (m_bonds_made.find(b_type) == m_bonds_made.end())
            m_bonds_made[b_type] = 0; //resetting bonds made count in the beginning.
        //setting some class variables for each bond type
        if (m_summed_bps.find(b_type) == m_summed_bps.end())
            m_summed_bps[b_type] = 0.0;
        //populate the pair that stores the bond type and their participants as <bond_type,participant1 type, participant2 type>
        if (m_bond_type.find(b_type) == m_bond_type.end())
            m_bond_type[b_type] = std::make_pair(A_type,B_type);
        else
            m_exec_conf->msg->warning() << "Bond type: " << b_type << "is already defined!";
        //search for existing entries for particle A. If found display a warning, else add it into the dictionary of particle type
        if (m_functional_groups.find(A_type) == m_functional_groups.end())
            m_functional_groups[A_type] = A_fun_groups;
        else
            m_exec_conf->msg->warning() << "Particle type: " << A << "with " << A_fun_groups << "functional groups is already defined!";
        //search for existing entries for particle B
        if (m_functional_groups.find(B_type) == m_functional_groups.end())
            m_functional_groups[B_type] = B_fun_groups;
        else
            m_exec_conf->msg->warning() << "Particle type: " << B << "with " << B_fun_groups << "functional groups is already defined!";
        printf("DyBondUpdater::set_params bond type:%s, A %s, A_fun_group %d, B:%s, B_fun_groups:%d, Ea:%f, alpha:%f\n",bond_type.c_str(),A.c_str(), A_fun_groups,B.c_str(),B_fun_groups,Ea,alpha);

        //create a dictionary of particle types as key and the number of particles of that type as value.
        std::map<unsigned int,unsigned int> particle_cnt;
        for (auto& ptype : m_functional_groups)
            particle_cnt[ptype.first] = 0;
        //now iterate through all the particles in the system and count the number of those types of particles.
        //note that there might be other particle types in the system that we may not be interested in.
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        unsigned int total_particles = (unsigned int) m_pdata->getN();
        for (unsigned int i=0;i<total_particles;i++)
            {
            int type = __scalar_as_int(h_pos.data[i].w);
            if (particle_cnt.find(type) != particle_cnt.end())
                particle_cnt[type]++;
            }
        //now loop through the bond types and count total number of bonds possible for each bond type
        //WARNING! This algorithm breaks down when the same particle type takes part in two bond types.
        //TODO: Fix this problem with the algorithm.
        unsigned int total_possible_bonds = 0;
        for(auto& btype : m_bond_type)
            {
            unsigned int typeA = std::get<0>(btype.second);
            unsigned int typeB = std::get<1>(btype.second);
            unsigned int num_bondsA = particle_cnt[typeA]*m_functional_groups[typeA];
            unsigned int num_bondsB = particle_cnt[typeB]*m_functional_groups[typeB];
            unsigned int max_possible_bonds = (num_bondsA>num_bondsB) ? num_bondsB : num_bondsA;
            m_possible_bonds[btype.first] = max_possible_bonds;
            m_max_bonds_per_attempt[btype.first] = max_possible_bonds*percent_bonds_per_step/100.0;//bond_attempts;
            printf("Total %s type bonds possible: %d, %f bonds will be attempted per bond step\n",
                   bond_type.c_str(),max_possible_bonds,m_max_bonds_per_attempt[btype.first]);
            total_possible_bonds += max_possible_bonds;
            }
        printf("Total bonds possible: %d\n",total_possible_bonds);
        init_dictionaries_from_system(b_type);
        printf("Percentage of %s type bonds made at the begining of simulation: %f\n",bond_type.c_str(),get_bond_percent(b_type));
        }
        else
        {
        m_exec_conf->msg->warning() << "The set_params function of dybond updater was used more than once."
                                    << " As of now, the dybond updater is only tested for one bond type."
                                    << " Hence only the first call to set_params has been used. Others were ignored\n";
        }
    }

/*! \param bond_type: bond type to search from system and populate dictionaries
    This is done so that if the initial condition contains some bonds of type that
    are being made by the DyBondUpdater, it needs to know those bonds so that we
    do not make duplicate bonds and the logged quantities reflect the actual numbers.
    This is requied because we use two dictionaries to speed up look up during the actual simulation.
    They are
    1) m_bond_data = { bond type : number of bonds of this type }
    2) m_rank_dict = { particle id : number of bonds made by this particle }
*/
void DyBondUpdater::init_dictionaries_from_system(unsigned int bond_type)
    {
    ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    unsigned int num_bonds = 0;
    // for each of the bonds
    const unsigned int size = (unsigned int)m_bond_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        auto curr_b_type = m_bond_data->getTypeByIndex(i);
        if(curr_b_type!=bond_type)
            continue;
        //printf("iterating bonds,i:%d, type:%d, bond type being checked:%d \n",i,curr_b_type,bond_type);
        // lookup the tag of each of the particles participating in the bond
        const BondData::members_t bond = m_bond_data->getMembersByIndex(i);
        assert(bond.tag[0] < m_pdata->getN());
        assert(bond.tag[1] < m_pdata->getN());

        unsigned int idx_a = h_rtag.data[bond.tag[0]];//provide tags to get the actual id
        unsigned int idx_b = h_rtag.data[bond.tag[1]];
        assert(idx_a <= m_pdata->getMaximumTag());
        assert(idx_b <= m_pdata->getMaximumTag());
        auto a_bond_rank = bond_rank(idx_a);
        auto b_bond_rank = bond_rank(idx_b);
        m_rank_dict[idx_a] = a_bond_rank + 1;
        m_rank_dict[idx_b] = b_bond_rank + 1;
        num_bonds++;
        }
    m_bonds_made[bond_type] = num_bonds;
    }
/*! Returns the number of times this particle has been bonded already
 * \params p_type: p_idx:  the particle index
*/
unsigned int DyBondUpdater::bond_rank(unsigned int p_idx)
    {
    //TODO: we could pre populate the dictionary before hand because all the particles will be bonded eventually
    unsigned int bond_rank = 0;
    if (m_rank_dict.find(p_idx) == m_rank_dict.end())
    {
    bond_rank = 0;
    m_rank_dict[p_idx] = bond_rank;
    }
    else
    {
    bond_rank = m_rank_dict[p_idx];
    }
    return bond_rank;
    }

/*! Checks if particle type "p_type" is a participating type in bond type "b_type"
 * \params b_type: the bond type
 *         p_type: the particle type
 */
bool DyBondUpdater::is_bondable_particle(unsigned int b_type, unsigned int p_type)
    { 
    return (m_bond_type[b_type].first == p_type || m_bond_type[b_type].second == p_type); 
    }

template <typename T>
std::vector<size_t> DyBondUpdater::sort_indexes(const std::vector<T> &v)
    {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
    }
//! Helper function to generate a [0..1] float
/*! \param rnd Random number generator to use
*/
static Scalar random01(std::mt19937& rnd)
    {
    unsigned int val = rnd();

    double val01 = ((double)val - (double)rnd.min()) / ( (double)rnd.max() - (double)rnd.min() );
    return Scalar(val01);
    }

/*! The main function in the updater code that gets called every period. 
 *  Computes the neighbourlist, temperature and uses particle information to make bonds with a probability.
    \param timestep Current time step of the simulation
*/
void DyBondUpdater::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("DyBondUpdater");
    bool nothing_to_do = true;
    for (auto& current_bond_type_entry : m_bond_type)//this is done to avoid computing nlist if unnecessary
        {
        if (!m_stop_bonding[current_bond_type_entry.first])
            nothing_to_do = false;
        }
    //printf("Inside DybonUpdater, timestep %d, nothing_to_do %d\n",timestep,nothing_to_do);
    if (!nothing_to_do)
        {
        // start by updating the neighborlist
        m_nlist->compute(timestep);
        // compute the current thermodynamic properties and get the temperature
        m_thermo->compute(timestep);
        Scalar curr_T = m_thermo->getTranslationalTemperature();
        // access the neighbor list
        ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);
        // access the particle data
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
        ArrayHandle< unsigned int > h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

        m_num_bonds_per_step = 0;
        //sanity check
        assert(m_bond_type.size() > 0);
        BoxDim box = m_pdata->getBox();
        //try bonding every type of bond requested by the user
        for (auto& current_bond_type_entry : m_bond_type)
            {
            auto curr_b_type = current_bond_type_entry.first;
            auto curr_b_pair = current_bond_type_entry.second;
            unsigned int total_particles = (unsigned int) m_pdata->getN();
            //printf("max bonds attempted per step for bond type %d will be %f.\n",curr_b_type,m_max_bonds_per_attempt[curr_b_type]);
            //printf("Bonding will stop after %f percent\n",m_stop_after_percent[curr_b_type]);
            Scalar bps = m_max_bonds_per_attempt[curr_b_type];
            unsigned int d_bps = 0;
            if (bps <1.0)
                {
                if (m_summed_bps[curr_b_type] < 1.0)
                    m_summed_bps[curr_b_type] += bps;
                if (m_summed_bps[curr_b_type] >= 1.0)
                    {
                    d_bps = round(m_summed_bps[curr_b_type]);
                    m_summed_bps[curr_b_type] = 0.0;
                    }
                }
            else
                d_bps = round(bps);
            //printf("bps:%f, d_bps:%x\n",bps,d_bps);
            for (unsigned int bnd_cnt=0;bnd_cnt<d_bps;bnd_cnt++)
                {
                if (!m_stop_bonding[curr_b_type])
                    {
                    unsigned int p_from_type = 9999;
                    unsigned int p_to_type = 9999;
                    unsigned int p_from_idx=0;
                    unsigned int p_from_tag=0;
                    //loop until a random particle is a bondable particle
                    while(!is_bondable_particle(curr_b_type, p_from_type))
                        {
                        p_from_idx = round(random01(m_rnd)*(total_particles-1));
                        p_from_tag = h_rtag.data[p_from_idx];
                        p_from_type = __scalar_as_int(h_pos.data[p_from_tag].w);// this is'nt the most efficient way to pick a bondable particle
                        }
                    //printf("Bond from particle id: %d timestep:%d\n",p_from_idx,timestep);

                    //check if the bond_from particle can bond
                    auto rank1 = bond_rank(p_from_idx);
                    if (m_functional_groups[p_from_type] > rank1)
                        {
                        //pick the complementary particle to bond to
                        auto p1_type = std::get<0>(curr_b_pair);
                        auto p2_type = std::get<1>(curr_b_pair);
                        p_to_type = (p_from_type==p1_type) ? p2_type : p1_type;
                        // sanity check
                        assert(p_from_type < m_pdata->getNTypes());
                        // loop over all of the neighbors of this particle and try bonding.
                        // If a bond is made successfully, break!
                        const unsigned int size = (unsigned int)h_n_neigh.data[p_from_tag];//p_from_idx];
                        const unsigned int head_i = h_head_list.data[p_from_tag];//p_from_idx];
                        //if (size>0)
                        //    printf("number of neighbors for %d : %d\n",p_from_idx,size);
                        //else
                        //    printf("no neighbours found\n");
                        std::vector<Scalar> distances;
                        std::vector<unsigned int> p_to_idxs;
                        std::vector<unsigned int> rank2s;
                        for (unsigned int j = 0; j < size; j++)
                            {
                            unsigned int p_to_tag = h_nlist.data[head_i + j];
                            auto p_to_idx = h_tag.data[p_to_tag];
                            unsigned int p_temp_type = __scalar_as_int(h_pos.data[p_to_tag].w);
                            vec3<Scalar> pi(h_pos.data[p_from_tag].x, h_pos.data[p_from_tag].y, h_pos.data[p_from_tag].z);
                            vec3<Scalar> pj(h_pos.data[p_to_tag].x, h_pos.data[p_to_tag].y, h_pos.data[p_to_tag].z);
                            vec3<Scalar> dxScalar(pj - pi);
                            // apply periodic boundary conditions (FLOPS: 9 (worst case: first branch is missed, the 2nd is taken and the add is done)
                            dxScalar = vec3<Scalar>(box.minImage(vec_to_scalar3(dxScalar)));
                            Scalar dist = sqrt(dxScalar.x*dxScalar.x + dxScalar.y*dxScalar.y + dxScalar.z*dxScalar.z);
                            auto rank2 = bond_rank(p_to_idx);
                            auto within_cutoff = (dist < m_rcut);
                            auto bondable_type = (p_temp_type == p_to_type);
                            auto bondable_rank = (m_functional_groups[p_to_type] > rank2);
                            //printf("dist: %f, rcut: %f\n",dist,m_rcut);
                            //printf("cutoff %d, type %d, rank %d\n",within_cutoff,bondable_type,bondable_rank);
                            if (within_cutoff && bondable_type && bondable_rank)
                                {
                                //printf("Adding into filtered array: dist %f\n",dist);
                                distances.push_back(dist);
                                p_to_idxs.push_back(p_to_idx);
                                rank2s.push_back(rank2);
                                }
                            }
                        /*
                        if (distances.size() > 1)
                            {
                            printf("sorted distances: %lu timestep:%d\n",distances.size(),timestep);
                            for (auto j: sort_indexes(distances))
                                {
                                printf("dist: %f, rcut: %f\n",distances[j],m_rcut);
                                }
                            }
                        */
                        for (auto j: sort_indexes(distances))
                            {
                            //printf("dist: %f, rcut: %f\n",distances[j],m_rcut);
                            auto rank2 = rank2s[j];
                            auto p_to_idx = p_to_idxs[j];
                            Scalar alpha = 1.0;
                            if (rank1>=1 || rank2>=1)//for now if either of the particles have been bonded before, increase the activation energy
                                alpha = m_alpha;
                            Scalar mb_stats = exp(-m_Ea*alpha/curr_T);
                            Scalar temp_rand = random01(m_rnd);
                            //printf("mb_stats: %f rand %f m_Ea:%f curr_T:%f\n",mb_stats,temp_rand,m_Ea,curr_T);
                            if (mb_stats>temp_rand)
                                {
                                //printf("neighbour of %d: %d (%f)\n",p_from_idx,p_to_idx, dist);
                                //printf("p1:%f,%f,%f, p1:%f,%f,%f\n",x1,y1,z1,x2,y2,z2);
                                //printf("Bond from %d (%d) to %d (%d) (%f)@ timestep %d\n",p_from_idx,p_from_type,p_to_idx,
                                m_bond_data->addBondedGroup(Bond(curr_b_type,p_from_idx,p_to_idx));
                                m_num_bonds_per_step++;
                                //m_nlist->addExclusionsFromBonds();
                                if (m_exclude_from_nlist)
                                    m_nlist->addExclusion(p_from_idx,p_to_idx);
                                //m_nlist->countExclusions();
                                m_rank_dict[p_from_idx] += 1;
                                m_rank_dict[p_to_idx] += 1;
                                m_bonds_made[curr_b_type]++;
                                double curr_cure_percent = 100.*m_bonds_made[curr_b_type]/m_possible_bonds[curr_b_type];
                                double sap = m_stop_after_percent[curr_b_type];
                                //printf("curr_cure_percent %f bonds made:%d stop_after %f\n",curr_cure_percent,m_bonds_made[curr_b_type],sap);
                                //printf("stop bonding? %d\n",curr_cure_percent >= sap);
                                if (curr_cure_percent >= sap)
                                    {
                                    m_callback[curr_b_type](timestep, curr_cure_percent,0);
                                    //auto it = m_bond_type.find(curr_b_type);
                                    //m_bond_type.erase(it);
                                    //bnd_cnt = m_max_bonds_per_attempt[curr_b_type];//a trick to get out of the current loop.
                                    m_stop_bonding[curr_b_type] = true;
                                    }
                                break;
                                }
                            }
                        }
                    }
                }
            }
        //printf("m_num_bonds_per_step %d \n",m_num_bonds_per_step);
        if (m_num_bonds_per_step > 0 && m_enable_enthalpy==true)
            {
            Scalar Ndelta_T = addHeat(timestep, m_num_bonds_per_step, curr_T);
            if (Ndelta_T !=0)
                {
                for (auto& current_bond_type_entry : m_bond_type)
                    {
                    auto curr_b_type = current_bond_type_entry.first;
                    auto curr_cure_percent = 100 * m_bonds_made[curr_b_type] / m_possible_bonds[curr_b_type];
                    m_callback[curr_b_type](timestep, curr_cure_percent, Ndelta_T);
                    }
                }

            }
        }
    if (m_prof) m_prof->pop();
    }

Scalar DyBondUpdater::addHeat(unsigned int timestep,
                            unsigned int num_bonds_created,
                            Scalar current_temperature)
    {
    //printf("num bonds created %u\n",num_bonds_created);
    Scalar Ndelta_T = m_deltaT * num_bonds_created;
    if (current_temperature < 1e-3)
        {
        m_exec_conf->msg->notice(2) << "update.DyBondUpdater: cannot scale a 0 translational temperature to anything but 0, skipping this step" << std::endl;
        }
    else
        {
        // calculate a fraction to scale the momenta by
        Scalar new_T = current_temperature + Ndelta_T;
        Scalar fraction = sqrt(new_T / current_temperature);
        //printf("fraction: %f, current: %f, new: %f\n",fraction,current_temperature,new_T);
        // scale the free particle velocities
        if (fraction != 0)
            {
            assert(m_pdata);
                {
                ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);

                for (unsigned int i = 0; i < m_pdata->getN(); i++)
                    {
                    h_vel.data[i].x *= fraction;
                    h_vel.data[i].y *= fraction;
                    h_vel.data[i].z *= fraction;
                    }
                }
            }
            else
                {
                m_exec_conf->msg->notice(2) << "update.DyBondUpdater: cannot scale translational  temperature by 0"<<std::endl;
                }
        }

    Scalar cur_rot_temp = m_thermo->getRotationalTemperature();
    if (! std::isnan(cur_rot_temp))
        {
        printf("rescaling rotational velocity\n");
        // only rescale if we have rotational degrees of freedom
        if (cur_rot_temp < 1e-3)
            {
            m_exec_conf->msg->notice(2) << "update.temp_rescale: cannot scale a 0 rotational temperature to anything but 0, skipping this step" << std::endl;
            }
        else
            {
            // calculate a fraction to scale the momenta by
            Scalar new_T = current_temperature + Ndelta_T;
            Scalar fraction = sqrt(new_T / cur_rot_temp);

            // scale the free particle velocities
            assert(m_pdata);
                {
                ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);

                for (unsigned int i = 0; i < m_pdata->getN(); i++)
                    {
                    h_angmom.data[i].x *= fraction;
                    h_angmom.data[i].y *= fraction;
                    h_angmom.data[i].z *= fraction;
                    }
                }

            }
        }
    return Ndelta_T;
    }

std::vector< std::string > DyBondUpdater::getProvidedLogQuantities()
    {
    std::vector<std::string> ret;
    for(auto& btype : m_bond_type)
        {
        auto bond_type_str = m_bond_data->getNameByType(btype.first);
        ret.push_back("bond_percent("+bond_type_str+")");
        ret.push_back("bonds_per_step("+bond_type_str+")");
        }
    m_loggable_quantities = ret; //This is a deep copy. Any more loggable quantitites need to be appended.
    return ret;
    }

Scalar DyBondUpdater::get_bond_percent(unsigned int bond_type)
    {
    Scalar curr_cure_percent = (100.*m_bonds_made[bond_type]/m_possible_bonds[bond_type]);
    return curr_cure_percent;
    /*const unsigned int total_bonds = (unsigned int)m_bond_data->getN();
    unsigned int bond_cnt = 0;
    for (unsigned int i = 0; i < total_bonds; i++)
        {
        unsigned int this_btype = m_bond_data->getTypeByIndex(i);
        //printf("bond (%d) %d - %d\n",btype,bond.tag[0],bond.tag[1]);
        if (this_btype == bond_type)
            bond_cnt++;
        }
    //printf("total bonds: %d, bond count: %d\n",total_bonds,bond_cnt);
    Scalar bond_percent = bond_cnt*100./m_possible_bonds[bond_type];
    //printf("bond_percent %f\n",bond_percent);
    return bond_percent;*/
    }

Scalar DyBondUpdater::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (std::find(m_loggable_quantities.begin(),m_loggable_quantities.end(),quantity) != m_loggable_quantities.end())
        {
        unsigned int first = quantity.find('(')+1;
        unsigned int last = quantity.find_last_of(')');
        std::string this_b_name = quantity.substr(first,last-first);
        //printf("log bond name: %s\n",this_b_name.c_str());
        std::string this_quantity_name = quantity.substr(0,first-1);
        //printf("getting log quantity %s\n",this_quantity_name.c_str());
        if (this_quantity_name == "bond_percent")
            return get_bond_percent(m_bond_data->getTypeByName(this_b_name));
        else if (this_quantity_name == "bonds_per_step")
            return m_num_bonds_per_step;
        }
    else
        {
        m_exec_conf->msg->error() << "update.dybond updater: " << quantity
                                  << " is not a valid log quantity." << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    return Scalar(0.0);
    }

/* Export the CPU updater to be visible in the python module
 */
void export_DyBondUpdater(pybind11::module& m)
    {
    pybind11::class_<DyBondUpdater, std::shared_ptr<DyBondUpdater> >(m, "DyBondUpdater", pybind11::base<Updater>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
        std::shared_ptr<NeighborList>,
        std::shared_ptr<ComputeThermo> >())
        .def("set_params", &DyBondUpdater::set_params)
    ;
    }

// ********************************
// here follows the code for DyBondUpdater on the GPU
// WARNING: This code is not complete! 
#ifdef ENABLE_CUDA

/*! \param sysdef System to zero the velocities of
*/
DyBondUpdaterGPU::DyBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<NeighborList> nlist,
                                   std::shared_ptr<ComputeThermo> thermo)
        : DyBondUpdater(sysdef, nlist,thermo)


    {
    m_exec_conf->msg->warning() << "DyBondUpdaterGPU is not implemented!";
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void DyBondUpdaterGPU::update(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile
    if (m_prof) m_prof->push("DyBondUpdater");

    // access the particle data arrays for reading on the GPU
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    
    // Access the bond table for reading
    ArrayHandle<BondData::members_t> d_gpu_bondlist(this->m_bond_data->getGPUTable(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int > d_gpu_n_bonds(this->m_bond_data->getNGroupsArray(), access_location::device, access_mode::read);

    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(this->m_nlist->getHeadList(), access_location::device, access_mode::read);         
    // call the kernel devined in DyBondUpdater.cu
    gpu_zero_velocities(d_pos.data, 
                        m_pdata->getN(),
                        d_gpu_bondlist.data,
                        m_bond_data->getGPUTableIndexer().getW(),
                        d_gpu_n_bonds.data,
                        m_bond_data->getNTypes(),
                        d_n_neigh.data,
                        d_nlist.data,
                        d_head_list.data,
                        this->m_nlist->getNListArray().getPitch());
	
    printf("in Dybondupdated GPU::update. Time step:%d\n",timestep);
    // check for error codes from the GPU if error checking is enabled
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop();
    }

/* Export the GPU updater to be visible in the python module
 */
void export_DyBondUpdaterGPU(pybind11::module& m)
    {
    pybind11::class_<DyBondUpdaterGPU, std::shared_ptr<DyBondUpdaterGPU> >(m, "DyBondUpdaterGPU", pybind11::base<DyBondUpdater>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
        std::shared_ptr<NeighborList>,
        std::shared_ptr<ComputeThermo> >())
    ;
    }

#endif // ENABLE_CUDA	
