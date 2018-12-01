from epoxpy.epoxy_simulation import EpoxySimulation
from epoxpy.lib import A, B, C, C10
import hoomd
import hoomd.dybond_plugin as db
from hoomd import md
import os
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import Counter
import epoxpy.common as cmn
from hoomd import variant
import mbuild as mb
import epoxpy.init as my_init
from epoxpy.utils import Angles


class ABCTypeEpoxySimulation(EpoxySimulation, metaclass=ABCMeta):
    """Simulations class for setting initial condition and force field specific to the ABC coarse grained Epoxy blend.
          This simulation consists of three particle types (A, B and C). A, B and C particles are created in the
          ratio 10, 20 and 2 by default
          The force fields used are Dissipative Particle Dynamics (DPD) and Harmonic potential

          sim_name : name of simulation
          mix_time : number of time steps to run the equilibration (NVE simulation to get a nicely "shaken" phase)
          md_time  : number of time steps to run the molecular dynamics simulation (NVE)
          mix_kt   : temperature at which the mixing should be performed (during this time, we do an NVT)
          temp_prof: Dictionary of temperature and time which specifies the temperature profile to maintain during the
                     md_time. If md_time exceeds the last time step mentioned in the temp_prof, the last temperature is
                     maintained. If md_time is lesser than the last time step in the profile, the simulation ends without
                     completing the prescribed profile.
          log_write: time interval with which to write the log file
          dcd_write: time interval with which to write the dcd file
          num_a    : number of A particles created.
          num_b    : number of B particles created.
          num_C    : number of C particles created.
          n_mul    : multiplying factor for the number of A, B and C particles to be created.
          gamma    : diffusive drag coefficient
          stop_after_percent: stops the dybond plugin after reaching this cure percent
          percent_bonds_per_step: percentage of possible bonds that will be made per bond_period timesteps
          AB_bond_const: the harmonic bond coefficient used for AB bonds
          AB_bond_dist:  the equilibrium bond distance for AB
          CC_bond_const: the harmonic bond coefficient used for CC bonds
          CC_bond_dist:  the equilibrium bond distance for CC
          CC_bond_angle_const: the cosine squared bond angle coefficient used for CC bonds
          CC_bond_angle:  the equilibrium bond angle for CC in degrees
          output_dir: default is the working directory
          bond     : boolean value denoting whether to run the bonding routine for A's and B's
          bond_period: time interval between calls to the bonding routine
       """

    def __init__(self,
                 sim_name,
                 mix_time,
                 mix_kt,
                 temp_prof,
                 num_a=10,
                 num_b=20,
                 num_c10=2,
                 n_mul=1.0,
                 gamma=4.5,
                 stop_after_percent=100.0,
                 percent_bonds_per_step=0.0025,
                 AB_bond_const=100,
                 CC_bond_const=100,
                 AB_bond_dist=1,
                 CC_bond_dist=1,
                 shrink_time=1e6,
                 shrinkT=2.0,
                 CC_bond_angle=None,
                 CC_bond_angle_const=None,
                 max_a_bonds=4,
                 max_b_bonds=2,
                 *args,
                 **kwargs):
        EpoxySimulation.__init__(self,
                                 sim_name,
                                 mix_time,
                                 mix_kt,
                                 temp_prof,
                                 *args,
                                 **kwargs)
        self.num_a = int(num_a * n_mul)
        self.num_b = int(num_b * n_mul)
        self.num_c10 = int(num_c10 * n_mul)
        if self.num_c10 < 1:
            self.num_c10 = 1  # we throw in one toughener to allow using bonded potentials
        self.max_a_bonds = max_a_bonds
        self.max_b_bonds = max_b_bonds
        self.n_mul = n_mul
        self.group_a = None
        self.group_b = None
        self.group_c = None
        self.AB_bond_const=AB_bond_const
        self.CC_bond_const=CC_bond_const
        self.AB_bond_dist=AB_bond_dist
        self.CC_bond_dist=CC_bond_dist
        self.CC_bond_angle=CC_bond_angle
        self.CC_bond_angle_const=CC_bond_angle_const
        self.gamma = gamma
        self.stop_after_percent = stop_after_percent
        self.percent_bonds_per_step = percent_bonds_per_step
        self.shrink_time = shrink_time
        self.shrinkT = shrinkT

        if self.bond:
            total_a_bonds = self.num_a*self.max_a_bonds
            total_b_bonds = self.num_b*self.max_b_bonds
            if total_a_bonds != total_b_bonds:
                print('ERROR: total_a_bonds:{}, total_b_bonds:{}'.format(total_a_bonds,total_b_bonds))
                print('ERROR: num_a:{}, num_b:{}, max_a_bonds:{},max_b_bonds:{}'.format(self.num_a,
                                                                                        self.num_b,
                                                                                        self.max_a_bonds,
                                                                                        self.max_b_bonds))
                raise ValueError('ABCTypeEpoxySimulation has a non stoichiometric configuration.\
                                  Please check num_a, max_a_bonds,num_b and max_b_bonds')

    def get_log_quantities(self):
        log_quantities = super().get_log_quantities()
        if self.dybond_updater is not None:
            log_quantities.append("bond_percent(A-B)")
            log_quantities.append("bonds_per_step(A-B)")
        return log_quantities

    def get_msd_groups(self):
        self.group_a = hoomd.group.type(name='a-particles', type='A')
        self.group_b = hoomd.group.type(name='b-particles', type='B')
        self.group_c = hoomd.group.type(name='c-particles', type='C')
        msd_groups = [self.group_a, self.group_b, self.group_c]
        return msd_groups

    def get_system_from_file(self, file_path, use_time_step_from_file):
        if use_time_step_from_file:
            time_step = None
        else:
            time_step = 0  # start simulation from start.
        if file_path.endswith('.hoomdxml'):
            system = hoomd.deprecated.init.read_xml(file_path, time_step=time_step, wrap_coordinates=True)
        elif file_path.endswith('.gsd'):
            #raise ValueError('Reading the most recent frame from gsd file is not yet implemented!')
            system = hoomd.init.read_gsd(file_path, frame=-1)
        else:
            raise ValueError('No such file as {} exist on disk!'.format(file_path))
        return system

    def initialize_system_from_file(self, file_path, use_time_step_from_file=True):
        self.system = self.get_system_from_file(file_path, use_time_step_from_file)
        if self.system is None:
            raise ValueError('get_system_from_file did not return a valid system object!')
        snapshot = self.system.take_snapshot(bonds=True)
        snapshot.bonds.types = ['C-C', 'A-B']  # this information is not stored in the hoomdxml. So need to add it in.
        self.system.restore_snapshot(snapshot)

    def set_initial_structure(self):
        print('========INITIAIZING STRUCTURE==========')
        desired_box_volume = ((A.mass*self.num_a) + (B.mass*self.num_b) + (C10.mass*self.num_c10)) / self.density
        desired_box_dim = (desired_box_volume ** (1./3.))
        reduced_density = self.density/10
        ex_box_vol = ((A.mass * self.num_a) + (B.mass * self.num_b) + (C10.mass * self.num_c10)) / reduced_density
        expanded_box_dim = (ex_box_vol ** (1. / 3.))
        half_L = expanded_box_dim/2
        box = mb.Box(mins=[-half_L, -half_L, -half_L], maxs=[half_L, half_L, half_L])
        if self.old_init:
            print("\n\n ===USING OLD INIT=== \n\n")
            As = my_init.Bead(btype="A", mass=A.mass)
            Bs = my_init.Bead(btype="B", mass=B.mass)
            C10s = my_init.PolyBead(btype="C", mass=1.0, N=10)# Hardcode C10, with mon-mass 1.0
            snap = my_init.init_system({As: int(self.num_a),
                                        Bs: int(self.num_b),
                                        C10s: int(self.num_c10)},
                                       self.density/10)
            system = hoomd.init.read_snapshot(snap)
        else:
            print("\n\n ===USING MBUILD INIT=== \n\n")
            if self.shrink is False:
                print('shrink=False is deprecated.')
            print('Packing {} A particles, {} B particles and {} C10s ..'.format(self.num_a,
                                                                                 self.num_b,
                                                                                 self.num_c10))
            mix_box = mb.packing.fill_box([A(), B(), C10()],
                                          [self.num_a, self.num_b, self.num_c10],
                                          box=box)  # ,overlap=0.5)

            if self.init_file_name.endswith('.hoomdxml'):
                mix_box.save(self.init_file_name, overwrite=True, ref_distance=.1)
            elif self.init_file_name.endswith('.gsd'):
                mix_box.save(self.init_file_name, write_ff=False, overwrite=True)

            if self.init_file_name.endswith('.hoomdxml'):
                system = hoomd.deprecated.init.read_xml(self.init_file_name, wrap_coordinates=True)
            elif self.init_file_name.endswith('.gsd'):
                system = hoomd.init.read_gsd(self.init_file_name)

            print('Initial box dimension: {}'.format(system.box.dimensions))

            snapshot = system.take_snapshot(bonds=True)
            for p_id in range(snapshot.particles.N):
                p_types = snapshot.particles.types
                p_type = p_types[snapshot.particles.typeid[p_id]]
                if p_type == 'A':
                    snapshot.particles.mass[p_id] = A.mass
                if p_type == 'B':
                    snapshot.particles.mass[p_id] = B.mass
                if p_type == 'C':
                    snapshot.particles.mass[p_id] = C.mass
            print(snapshot.bonds.types)
            snapshot.bonds.types = ['C-C', 'A-B']
            system.restore_snapshot(snapshot)

        self.nl = self.get_non_bonded_neighbourlist()
        if self.nl is None:
            raise Exception('Neighbourlist is not set')
        self.setup_force_fields(stage=cmn.Stages.MIXING)
        size_variant = variant.linear_interp([(0, system.box.Lx), (self.shrink_time, desired_box_dim)])
        md.integrate.mode_standard(dt=self.mix_dt)
        md.integrate.langevin(group=hoomd.group.all(),
                              kT=self.shrinkT,
                              seed=1223445)  # self.seed)
        hoomd.update.box_resize(L=size_variant)
        hoomd.run(self.shrink_time)
        snapshot = system.take_snapshot()
        print('Initial box dimension: {}'.format(snapshot.box))
        return system

    def add_angles(self):
        snapshot = self.system.take_snapshot(all=True)
        if snapshot.angles.N == 0 and self.CC_bond_angle is not None:
            angles = Angles()
            ccc_angles = angles.get_angles_for_linear_chains(snapshot, 'C')
            snapshot.angles.types = ['C-C-C']
            for ccc_angle in ccc_angles:
                n_angles = snapshot.angles.N
                snapshot.angles.resize(n_angles + 1)
                snapshot.angles.group[n_angles] = ccc_angle
                snapshot.angles.typeid[n_angles] = 0  # we know C-C-C bond angle's id is 0
            self.system.restore_snapshot(snapshot)

    @abstractmethod
    def get_non_bonded_neighbourlist(self):
        pass

    @abstractmethod
    def setup_force_fields(self, stage):
        pass

    @abstractmethod
    def setup_integrator(self, stage):
        pass

    @abstractmethod
    def exclude_bonds_from_nlist(self):
        pass

    def setup_mixing_run(self):
        # Mix Step/MD Setup
        print('==============Setting up MIXING run=================')
        self.nl = self.get_non_bonded_neighbourlist()
        if self.nl is None:
            raise Exception('Neighbourlist is not set')
        self.setup_force_fields(stage=cmn.Stages.MIXING)
        self.setup_integrator(stage=cmn.Stages.MIXING)

    def stop_dybond_updater(self, timestep):
        if self.stop_dybond_updater_callback is not None:
            self.dybond_updater.disable() # first stop the updater
            self.stop_dybond_updater_callback.disable() # now stop the callback.
        else:
            hoomd.context.msg.warning('Call back for stopping the bonding is not set!')

    def setup_md_run(self):
        self.nl = self.get_non_bonded_neighbourlist()
        if self.nl is None:
            raise Exception('Neighbourlist is not set')
        self.setup_force_fields(stage=cmn.Stages.CURING)
        self.setup_integrator(stage=cmn.Stages.CURING)
        if self.bond is True:
            self.dybond_updater = db.update.dybond(self.nl, group=hoomd.group.all(), period=self.bond_period)
            print('#######################{}##########################'.format(type(self.exclude_bonds_from_nlist())))
            self.dybond_updater.set_params(bond_type='A-B',
                                           A='A',
                                           A_fun_groups=self.max_a_bonds,
                                           B='B',
                                           B_fun_groups=self.max_b_bonds,
                                           Ea=self.activation_energy,
                                           rcut=self.bond_radius,
                                           alpha=self.sec_bond_weight,
                                           percent_bonds_per_step=self.percent_bonds_per_step,
                                           stop_after_percent=self.stop_after_percent,
                                           callback=self.dybond_updater_callback,
                                           exclude_from_nlist=self.exclude_bonds_from_nlist(),
                                           enable_rxn_enthalpy=self.enable_rxn_enthalpy,
                                           deltaT=self.deltaT)

            if self.stop_bonding_after is not None:
                self.stop_dybond_updater_callback = hoomd.analyze.callback(callback=self.stop_dybond_updater,
                                                                           period=self.stop_bonding_after)

    def total_possible_bonds(self):
        if self.num_b * self.max_b_bonds > self.num_a * self.max_a_bonds:
           possible_bonds = (self.num_a * self.max_a_bonds)
        else:
            possible_bonds = (self.num_b * self.max_b_bonds)
        return possible_bonds

    def get_curing_percentage(self):
        n_bonds = 0
        if self.system is not None:
            snapshot = self.system.take_snapshot(bonds=True)
            n_bonds = len(snapshot.bonds.group) - (self.num_c10 * 9)
        possible_bonds = self.total_possible_bonds()
        bond_percent = (n_bonds / possible_bonds) * 100.
        #print('possible bonds:{}, bonds made:{}, cure percent: {}'.format(possible_bonds, n_bonds, bond_percent))
        return bond_percent

    '''deprecated: Used with freud and legacy bonding
    '''
    def calculate_curing_percentage(self, step):
        bond_percent = self.get_curing_percentage()
        self.curing_log.append((step, bond_percent))

        group_a_idx = []
        for p in self.group_a:
            group_a_idx.append(p.tag)
        #print(group_a_idx)
        group_b_idx = []
        for p in self.group_b:
            group_b_idx.append(p.tag)
        #print(group_b_idx)

        dic_vals = [self.bonding.rank_dict.get(k, 0) for k in group_a_idx]
        keys = (list(Counter(dic_vals).keys()))
        values = (list(Counter(dic_vals).values()))
        #print(keys)
        #print(values)
        row = np.zeros(5)
        for i in range(0, 4):
            if i + 1 in keys:
                row[i] = values[keys.index(i + 1)]
        row = [row]
        this_row = row[0]
        p_bonds = (100. * this_row[0]) / self.num_a
        s_bonds = (100. * this_row[1]) / self.num_a
        t_bonds = (100. * this_row[2]) / self.num_a
        q_bonds = (100. * this_row[3]) / self.num_a

        dic_vals = [self.bonding.rank_dict.get(k, 0) for k in group_b_idx]
        keys = (list(Counter(dic_vals).keys()))
        values = (list(Counter(dic_vals).values()))
        #print(keys)
        #print(values)
        if 0 in keys:
            primary_b = values[keys.index(0)]
        else:
            primary_b = 0
        primary_b = (100. * primary_b) / self.num_b

        row = [(step, bond_percent, p_bonds, s_bonds, t_bonds, q_bonds, primary_b)]
        with open(os.path.join(self.output_dir, self.bond_rank_hist_file), 'ab') as f_handle:
            np.savetxt(f_handle, row)

    def reset_setpoint_temperature(self, timestep, deltaT):
        raise NotImplementedError('reset_setpoint_temperature not implemented for', self)
