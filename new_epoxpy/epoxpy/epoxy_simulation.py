import os
from epoxpy.simulation import Simulation
import hoomd
from hoomd import md
from hoomd import deprecated
from hoomd import dump
from abc import ABCMeta, abstractmethod
import numpy.random as rd
import numpy as np
import errno
import epoxpy.common as cmn


class EpoxySimulation(Simulation, metaclass=ABCMeta):
    """Common base class for all epoxy simulations.
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
          bond     : boolean value denoting whether to run the bonding routine for A's and B's
          bond_period: time interval between calls to the bonding routine
          bond_radius: the minimum distance between two candidate particles for bonding
          enable_rxn_enthalpy: enables capturing enthalpy of reaction. If true, a rescale thermostatting is applied to
                               the system to increase the overall system temperature by delta_T.
          deltaT: the change in temperature per reaction. (this method is being evaluated)
          kwargs: These are parameters which might not be necessary from a user perspective. Includes parameters for
          backward compatibility also.
                legacy_bonding: boolean value indicating whether to use legacy bonding or frued bonding routine
                                Default is "False"
                exclude_mixing_in_output: boolean value indicating whether the initial mixing phase should appear
                                          in the output files. This may or may not be needed for analysis.
                                          Default is "False"
                init_file_name: Full path to the initial file being written to disk during initialization. Please do
                                not use this to initialize from an initial structure created externally. That is not
                                the current intention of this argument. For now, use this to switch the use of gsd file
                                vs. hoomdxml file. Why did I not just use one of them instead? Because hoomdxml is
                                human readable and is handy for debugging, but is deprecated. So if they remove support
                                its easy to switch to using gsd file format.
                shrink: boolean value indicating whether the initial structure will be shrunk to reach the specified
                        density. default is "True"
                shrink_time: number of timesteps to run hoomd to shrink the initial volume to desired density.
                             Default is 1 time step.
    """
    __metaclass__ = ABCMeta
    engine_name = 'HOOMD'

    def __init__(self,
                 sim_name,
                 mix_time,
                 mix_kt,
                 temp_prof,
                 log_write=100,
                 dcd_write=100,
                 bond=True,
                 bond_period=1e1,
                 bond_radius=1.0,
                 enable_rxn_enthalpy=False,
                 deltaT=0.0,
                 box=[3, 3, 3],
                 mix_dt=1e-2,
                 md_dt=1e-2,
                 density=1.0,
                 activation_energy=1.0,
                 sec_bond_weight=5.0,
                 stop_bonding_after=None,
                 **kwargs):
        Simulation.__init__(self, self.engine_name, **kwargs)
        self.simulation_name = sim_name
        self.mix_time = mix_time
        self.bond = bond
        self.bond_period = bond_period
        self.bond_radius = bond_radius
        self.enable_rxn_enthalpy = enable_rxn_enthalpy
        self.deltaT = deltaT
        self.mix_kT = mix_kt
        #final_time = temp_prof.get_total_sim_time()
        #md__total_time = final_time - mix_time
        self.temp_prof = temp_prof
        self.log_write = log_write
        self.dcd_write = dcd_write
        self.system = None
        self.mix_dt = mix_dt
        self.md_dt = md_dt
        self.density = density
        self.activation_energy = activation_energy
        self.sec_bond_weight = sec_bond_weight
        self.bonding = None
        self.bond_rank_hist_file = 'bond_rank_hist.log'
        self.bond_callback = None
        self.dybond_updater = None
        self.stop_bonding_after = stop_bonding_after
        self.stop_dybond_updater_callback = None
        self.nl = None

    def get_sim_name(self):
        return self.simulation_name

    @abstractmethod
    def initialize_system_from_file(self, init_file_path=None, use_time_step_from_file=True):
        pass

    @abstractmethod
    def setup_mixing_run(self):
        pass

    @abstractmethod
    def setup_md_run(self):
        pass

    @staticmethod
    def initialize_context():
        try:
            __IPYTHON__
            run_from_ipython = True
        except NameError:
            run_from_ipython = False
        if run_from_ipython:
            print('Initializing HOOMD in ipython')
            hoomd.context.initialize('--mode=cpu')
        else:
            hoomd.context.initialize()

    @staticmethod
    def silent_remove(filename):
        try:
            os.remove(filename)
        except OSError as e:  # this would be "except OSError, e:" before Python 2.6
            if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                raise  # re-raise exception if a different error occurred

    @abstractmethod
    def get_curing_percentage(self, step):
        pass

    @abstractmethod
    def calculate_curing_percentage(self, step):
        pass

    @abstractmethod
    def get_msd_groups(self):
        pass

    @abstractmethod
    def reset_setpoint_temperature(self, timestep, deltaT):
        pass

    def finalize_stage(self, stage):
        if stage == cmn.Stages.INIT:
            if self.init_file_name.endswith('.hoomdxml'):
                deprecated.dump.xml(group=hoomd.group.all(), filename=self.init_file_name, all=True)
            elif self.init_file_name.endswith('.gsd'):
                hoomd.dump.gsd(group=hoomd.group.all(), filename=self.init_file_name, overwrite=True, period=None)
        elif stage == cmn.Stages.MIXING:
            if self.mixed_file_name.endswith('.hoomdxml'):
                deprecated.dump.xml(group=hoomd.group.all(), filename=self.mixed_file_name, all=True)
            elif self.mixed_file_name.endswith('.gsd'):
                hoomd.dump.gsd(group=hoomd.group.all(), filename=self.mixed_file_name, overwrite=True, period=None)
        elif stage == cmn.Stages.CURING:
            deprecated.dump.xml(group=hoomd.group.all(),
                                filename=os.path.join(self.output_dir,'final.hoomdxml'),
                                all=True)
        self.dump_metadata()

    def dybond_updater_callback(self, timestep, bond_percent, deltaT):
        if deltaT > 0:
            self.reset_setpoint_temperature(timestep, deltaT)
        else:
            print('--------========= REACHED TARGET CURE PERCENT. STOPPED CURING AT {} @ TIME STEP:{} =========--------'
                  .format(bond_percent, timestep))

    def get_log_quantities(self):
        log_quantities = ["volume", "momentum", "potential_energy", "kinetic_energy", "temperature", "pressure"]
        return log_quantities

    def configure_outputs(self, stage=cmn.Stages.CURING):
        print('Configuring outputs. output_dir: {}'.format(self.output_dir))
        print('log_write: {} dcd_write: {}'.format(self.log_write, self.dcd_write))
        hoomd.meta.dump_metadata(filename=os.path.join(self.output_dir,
                                                       'metadata.json'), indent=2)
        deprecated.dump.xml(group=hoomd.group.all(),
                            filename=os.path.join(self.output_dir,
                                                  'start.hoomdxml'), all=True)
        quantities = self.get_log_quantities()

        print('quantities being logged:', quantities)
        if stage == cmn.Stages.CURING:
            log_period = self.log_write
            hoomd.analyze.log(filename=os.path.join(self.output_dir, 'out.log'),
                              quantities=quantities, period=log_period,
                              header_prefix='#', overwrite=False)
            dump.gsd(filename=os.path.join(self.output_dir, 'restart.gsd'), period=self.dcd_write,
                     group=hoomd.group.all(),
                     truncate=True)
            dump.dcd(filename=os.path.join(self.output_dir, 'traj.dcd'), period=self.dcd_write, overwrite=False)
            dump.gsd(filename=os.path.join(self.output_dir, 'data.gsd'), period=self.dcd_write, group=hoomd.group.all(),
                     overwrite=False, static=['attribute'])

            msd_groups = self.get_msd_groups()
            if msd_groups is not None:
                print('#### WARNING: no msd groups specified. Skipping msd logging!')
                if stage == cmn.Stages.MIXING:
                    deprecated.analyze.msd(groups=msd_groups, period=self.log_write, overwrite=True,
                                           filename=os.path.join(self.output_dir, 'msd.log'), header_prefix='#')

                else:
                    deprecated.analyze.msd(groups=msd_groups, period=self.log_write, overwrite=False,
                                           r0_file=self.mixed_file_name,
                                           filename=os.path.join(self.output_dir, 'msd.log'), header_prefix='#')
        else:
            log_period = self.log_write*10000
            hoomd.analyze.log(filename=os.path.join(self.output_dir, 'mixing.log'),
                              quantities=quantities, period=log_period,
                              header_prefix='#', overwrite=False)



        self.silent_remove(os.path.join(self.output_dir, self.bond_rank_hist_file))

    def run_mixing(self):
        hoomd.run(self.mix_time)

    @staticmethod
    def init_velocity(n, temp):
        v = rd.random((n, 3))
        v -= 0.5
        meanv = np.mean(v, 0)
        meanv2 = np.mean(v ** 2, 0)
        # fs = np.sqrt(3.*temp/meanv2)
        fs = np.sqrt(temp / meanv2)
        # print('scaling factor:{}'.format(fs))
        # print('v0:{}'.format(v))
        v = (v - meanv)  # shifts the average velocity of the simulation to 0
        v *= fs  # scaling velocity to match the desired temperature
        return v

    def initialize_snapshot_temperature(self, snapshot, temp):
        v = self.init_velocity(snapshot.particles.N, temp)
        snapshot.particles.velocity[:] = v[:]
        return snapshot

    def set_initial_particle_velocities(self, kT):
        snapshot = self.system.take_snapshot()
        snapshot = self.initialize_snapshot_temperature(snapshot, kT)
        self.system.restore_snapshot(snapshot)
        print('Reset the system temperature to {} kT after mixing'.format(kT))

    def run_md(self):
        first_target_temperature = self.temp_prof.temperature_profile[0][1]
        self.set_initial_particle_velocities(first_target_temperature)
        md_time = self.temp_prof.get_total_sim_time()
        print('md time: {}'.format(md_time))
        hoomd.run_upto(md_time+1)
        if self.profile_run:
            hoomd.run(int(md_time*0.1), profile=True)# run 10% of the simulation time to calculate performance
        if self.nl_tuning:
            print('-----------------Disabling bonding and starting neighbourlist tuning-------------------')
            if self.bond:
                if self.use_dybond_plugin:
                    self.dybond_updater.disable()
                else:
                    self.bond_callback.disable
            self.nl.tune(warmup=20000,
                         r_min=0.01,
                         r_max=2.00,
                         jumps=10,
                         steps=5000,
                         set_max_check_period=False)

        deprecated.dump.xml(group=hoomd.group.all(),
                            filename=os.path.join(self.output_dir,
                                                  'final.hoomdxml'), all=True)

    @abstractmethod
    def set_initial_structure(self):
        """
        Saves the initial structure to file. The file name to use is "self.init_file_name".
        Raises a value exception is this file is not found after this function call.
        :return:
        """
        pass

    @abstractmethod
    def get_system_from_file(self, file_path, use_time_step_from_file):
        """
        Reads in the input structure, initalizes the hoomd system object and returns it.
        :param file_path: path to input file
        :param use_time_step_from_file: Decides if the time step from the input file should be used or start from zero.
        :return: hoomd system object
        """
        pass

    @abstractmethod
    def add_angles(self):
        pass

    def initialize(self):
        print('Initializing {}'.format(self.simulation_name))
        if not os.path.exists(self.output_dir):
            print('Creating simulation folder: {}'.format(self.output_dir))
            os.makedirs(self.output_dir)

        # STEP 1: SET INITIAL CONDITION
        self.initialize_context()
        if self.ext_init_struct_path is None:
            self.system = self.set_initial_structure()
            if self.system is None:
                raise ValueError("set_initial_structure did not return a system object as expected!")
        else:
            print('Loading external initial structure : ', self.ext_init_struct_path)
            self.initialize_system_from_file(self.ext_init_struct_path)
            self.add_angles()

        self.finalize_stage(cmn.Stages.INIT)
        if not os.path.isfile(self.init_file_name):
            raise ValueError("finalize_stage did not save the init file ({}) as expected!".
                             format(self.init_file_name))

        # STEP 2: MIX
        del self.system  # needed for re initializing hoomd
        self.initialize_context()
        # initialize_system_from_file will make sure that the human readable bond types are added to the system
        self.initialize_system_from_file(self.init_file_name)
        self.setup_mixing_run()
        self.configure_outputs(stage=cmn.Stages.MIXING)
        self.run_mixing()
        self.finalize_stage(cmn.Stages.MIXING)
        if not os.path.isfile(self.mixed_file_name):
            raise ValueError("finalize_stage did not save the mixed file ({}) as expected!".
                             format(self.mixed_file_name))

    def run(self, sim_restart=False):
        del self.system
        self.initialize_context()
        if self.exclude_mixing_in_output:
            use_time_step_from_file = False
        else:
            use_time_step_from_file = True
        # Init from restart if it exists
        if sim_restart:
            self.initialize_system_from_file(self.restart_file_name, use_time_step_from_file=True)
        else:
            self.initialize_system_from_file(self.mixed_file_name, use_time_step_from_file=use_time_step_from_file)
        print('Running MD for {}'.format(self.simulation_name))
        self.setup_md_run()
        self.configure_outputs()

        hoomd.util.quiet_status()
        self.run_md()
        self.finalize_stage(cmn.Stages.CURING)

    def execute(self):
        print('Executing {}'.format(self.simulation_name))
        if os.path.isfile(self.restart_file_name):
            print("restarting simulation")
            self.run(sim_restart=True)
        else:
            self.initialize()
            if self.reset_random_after_initialize:
                import random
                random.seed(12345)
            self.run()
        print("Finished executing {}".format(self.simulation_name))
