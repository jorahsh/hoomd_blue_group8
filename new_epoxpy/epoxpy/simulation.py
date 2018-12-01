from abc import ABCMeta, abstractmethod
import json
import os
import time
import datetime
import numpy.random as rd


class Simulation(object, metaclass=ABCMeta):
    """Common base class for all simulations.
    output_dir: default is the working directory
    """
    __metaclass__ = ABCMeta

    simulation_name = 'Blank Simulation'
    engine_name = 'None'

    def __init__(self,
                 engine_name,
                 output_dir=os.getcwd(),
                 **kwargs):
        self.engine_name = engine_name
        self.output_dir = output_dir
        self.meta_data = dict()  # this meta data dictionary contains all the epoxpy related meta data

        # below are default developer arguments which can be set through kwargs in sub classes for testing.
        self.nl_tuning = False
        self.profile_run = False
        self.legacy_bonding = False
        self.use_dybond_plugin = True
        self.old_init = False
        self.exclude_mixing_in_output = True  # False  # PLEASE NOTE THAT THE TRAJECTORY CHANGES WHEN THIS IS CHANGED!!
        self.resume_file_name = os.path.join(self.output_dir, 'final.hoomdxml')
        self.init_file_name = os.path.join(self.output_dir, 'initial.hoomdxml')
        self.mixed_file_name = os.path.join(self.output_dir, 'mixed.hoomdxml')
        self.restart_file_name = os.path.join(self.output_dir, 'restart.gsd')
        self.shrink_time = 1.0
        self.shrink = True
        self.ext_init_struct_path = None
        self.log_curing = False
        self.curing_log_period = 1e5
        self.curing_log = []
        self.bond_rank_log = []
        self.DEBUG = True
        self.mbuild_seed = rd.randint(0, (2 ** 31) - 1)  # packmol complains if we pass it a seed larger than (2**31)-1
        self.meta_data['mbuild_seed'] = self.mbuild_seed

        # for tests which compare simulation result against a benchmark
        # please see issue 6 for more details
        # (https://bitbucket.org/cmelab/getting-started/issues/6/different-trajectory-obtained-when-using).
        self.reset_random_after_initialize = False

        print('kwargs passed into Simulation: {}'.format(kwargs))
        # setting developer variables through kwargs for testing.
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def get_sim_name(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def finalize_stage(self, stage):
        pass

    def dump_metadata(self, indent=4):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        self.meta_data['timestamp'] = st
        with open(os.path.join(self.output_dir, "epoxpy_metadata.json"), "w") as f:
            json.dump(self.meta_data, f, indent=indent, sort_keys=True)
