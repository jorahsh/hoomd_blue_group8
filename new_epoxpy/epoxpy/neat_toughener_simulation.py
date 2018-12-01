from epoxpy.epoxy_simulation import EpoxySimulation
import hoomd
import hoomd.dybond_plugin as db
import os
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import Counter
import epoxpy.common as cmn


class NeatToughenerSimulation(EpoxySimulation, metaclass=ABCMeta):
    """Simulations class for setting initial condition and force field specific to the coarse grained toughener solution.
          This simulation consists of one particle types (A).
          The force fields used can vary and depends on how the user inherits this class

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
          num_cxx    : number of C-mer particles created. The 'xx' denotes the number of beads in each chain.
          n_mul    : multiplying factor for the number of A, B and C particles to be created.
          output_dir: default is the working directory
          bond     : boolean value denoting whether to run the bonding routine for A's and B's
          bond_period: time interval between calls to the bonding routine
       """
    def __init__(self,
                 sim_name,
                 mix_time,
                 mix_kt,
                 temp_prof,
                 num_cXX=100,
                 cXX=10,
                 gamma=4.5,
                 CC_bond_const=100,
                 CC_bond_dist=1,
                 CC_bond_angle=None,
                 CC_bond_angle_const=None,
                 *args,
                 **kwargs):
        EpoxySimulation.__init__(self,
                                 sim_name,
                                 mix_time,
                                 mix_kt,
                                 temp_prof,
                                 *args,
                                 **kwargs)
        self.num_cXX = int(num_cXX)
        self.cXX = int(cXX)
        self.n_mul = num_cXX
        self.group_c = None
        self.CC_bond_const=CC_bond_const
        self.CC_bond_dist=CC_bond_dist
        self.CC_bond_angle = CC_bond_angle
        self.CC_bond_angle_const = CC_bond_angle_const
        self.gamma = gamma
        print('kwargs passed into ABCTypeEpoxySimulation: {}'.format(kwargs))
        # setting developer variables through kwargs for testing.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_log_quantities(self):
        log_quantities = super().get_log_quantities()
        return log_quantities

    def get_msd_groups(self):
        self.group_c = hoomd.group.type(name='c-particles', type='C')
        msd_groups = [self.group_c]
        return msd_groups

    def initialize_system_from_file(self, file_path, use_time_step_from_file=True):
        if use_time_step_from_file:
            time_step = None
        else:
            time_step = 0  # start simulation from start.

        if file_path.endswith('.hoomdxml'):
            self.system = hoomd.deprecated.init.read_xml(file_path, time_step=time_step, wrap_coordinates=True)
        elif file_path.endswith('.gsd'):
            raise ValueError('Reading the most recent frame from gsd file is not yet implemented!')
            self.system = hoomd.init.read_gsd(file_path, frame=0, time_step=time_step)
        else:
            raise ValueError('No such file as {} exist on disk!'.format(file_path))
        snapshot = self.system.take_snapshot(bonds=True)
        self.system.restore_snapshot(snapshot)

    def get_system_from_file(self, file_path, use_time_step_from_file):
        if use_time_step_from_file:
            time_step = None
        else:
            time_step = 0  # start simulation from start.
        if file_path.endswith('.hoomdxml'):
            system = hoomd.deprecated.init.read_xml(file_path, time_step=time_step, wrap_coordinates=True)
        elif file_path.endswith('.gsd'):
            raise ValueError('Reading the most recent frame from gsd file is not yet implemented!')
            system = hoomd.init.read_gsd(file_path, frame=0, time_step=time_step)
        else:
            raise ValueError('No such file as {} exist on disk!'.format(file_path))
        return system

    @abstractmethod
    def get_non_bonded_neighbourlist(self):
        pass

    @abstractmethod
    def setup_force_fields(self, stage):
        pass

    @abstractmethod
    def setup_integrator(self, stage):
        pass

    def add_angles(self):
        pass

    def get_curing_percentage(self):
        raise NotImplementedError('Not Applicable for neat toughener simulations')

    '''deprecated: Used with freud and legacy bonding
        '''
    def calculate_curing_percentage(self, step):
        raise NotImplementedError('Not Applicable for neat toughener simulations')

    def setup_mixing_run(self):
        # Mix Step/MD Setup
        print('==============Setting up MIXING run=================')
        self.nl = self.get_non_bonded_neighbourlist()
        if self.nl is None:
            raise Exception('Neighbourlist is not set')
        self.setup_force_fields(stage=cmn.Stages.MIXING)
        self.setup_integrator(stage=cmn.Stages.MIXING)

    def stop_dybond_updater(self, timestep):
        raise NotImplementedError('Bonding not applicable for neat toughener simulations')

    def setup_md_run(self):
        self.nl = self.get_non_bonded_neighbourlist()
        self.setup_force_fields(stage=cmn.Stages.CURING)
        self.setup_integrator(stage=cmn.Stages.CURING)

    def reset_setpoint_temperature(self, timestep, deltaT):
        raise NotImplementedError('reset_setpoint_temperature not implemented for', self)