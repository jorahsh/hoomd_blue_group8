from abc import ABCMeta, abstractmethod
from hoomd import variant
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


class TemperatureProfileBuilder(object):
    """Common base class for all TemperatureProfiles."""
    __metaclass__ = ABCMeta

    def __init__(self):
        self.temperature_profile = []

    @abstractmethod
    def get_profile(self):
        """
            Abstract method for TemperatureProfileBuilder.get_profile()
            Returns a hoomd.variant object
        """
        pass

    def get_figure(self):
        fig = plt.figure()
        x_val = [x[0] for x in self.temperature_profile]
        y_val = [x[1] for x in self.temperature_profile]

        plt.xlabel('Time')
        plt.ylabel('kT')
        plt.margins(x=0.1, y=0.1)
        plt.plot(x_val, y_val)
        plt.plot(x_val, y_val, 'or')
        return fig

    def get_total_sim_time(self):
        first_state_point = self.temperature_profile[0]
        last_state_point = self.temperature_profile[-1]
        return last_state_point[0]-first_state_point[0]

    def get_raw(self):
        return self.temperature_profile

    def set_raw(self, data):
        self.temperature_profile = data


class LinearTemperatureProfileBuilder(TemperatureProfileBuilder):
    """Builds a Linear Temperature Profile."""
    def __init__(self, initial_temperature, initial_time=0):
        TemperatureProfileBuilder.__init__(self)
        self.initial_temperature = initial_temperature
        self.temperature_profile.append((initial_time, initial_temperature))

    def add_state_point(self, ramp_time, desired_temperature):
        """
            Add a state point for the linear temperature profile. Adds ramp time to previous time or initial_time
        """
        last_state_point = self.temperature_profile[-1]
        new_state_point = ((last_state_point[0]+ramp_time), desired_temperature)
        if new_state_point[0] >= last_state_point[0]:
            self.temperature_profile.append(new_state_point)
        else:
            err_string = 'Inconsistent state point added. The new time should be greater or equal to previous time.' \
                         'Previous time: {}, new time: {}'.format(last_state_point[0], new_state_point[0])
            raise ValueError(err_string)

    def get_profile(self):
        """
            Returns a hoomd.variant object
        """
        profile = variant.linear_interp(points=self.temperature_profile, zero=0)
        return profile


class StepwiseQuenchTemperatureProfileBuilder(LinearTemperatureProfileBuilder):
    """Builds a Stepwise Temperature Profile for quenching."""
    def __init__(self, initial_temperature, final_temperature, quench_time, first_step_buffer_time, quench_interval=0.05, initial_time=0):
        TemperatureProfileBuilder.__init__(self)
        self.initial_temperature = initial_temperature
        self.temperature_profile.append((initial_time, initial_temperature))
        quenchTs=np.arange(initial_temperature,final_temperature-quench_interval,-quench_interval)
        for i,quenchT in enumerate(quenchTs):
            if i==0:
               self.add_state_point(quench_time+first_step_buffer_time,quenchT)
            else:
               self.add_state_point(1,quenchT)
               self.add_state_point(quench_time-1,quenchT)

