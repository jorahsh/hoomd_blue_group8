import random
import math
import mbuild as mb
import numpy as np
from epoxpy.lib import C


class C10(mb.Compound):
    """A coarse grained molecule consisting of 10 C particle which are coarse grained particles themselves.

        Note:
            Positions the first C particle at the prescribed position and translates the second C particle such that
            the 'down' port of the second C particle overlaps the 'up' port of the first C particle. This process is
            repeated for all subsequent particles. The position of the first particle and the orientation of it's
            ports determine the bounds of the molecule.

        Parameters
        ----------
        c1_pos : np.ndarray, shape=(3,), dtype=float
        The vector to translate the first C particle

        rotate_random : boolean
        rotates the first particle randomly if set to True

        """
    mass = 10.0

    def __init__(self, c1_pos=None, rotate_random=True):
        super(C10, self).__init__()
        num_particles = 10
        for index in range(num_particles):
            self.add(C(), label='C[$]')

        if c1_pos is not None:
            mb.Compound.translate(self['C'][0], c1_pos)

        if rotate_random is True:
            mb.Compound.rotate(self['C'][0], random.uniform(0, 2*math.pi), around=np.asarray([1, 0, 0]))
            mb.Compound.rotate(self['C'][0], random.uniform(0, 2*math.pi), around=np.asarray([0, 1, 0]))
            mb.Compound.rotate(self['C'][0], random.uniform(0, 2*math.pi), around=np.asarray([0, 0, 1]))

            #mb.rotate_around_x(self['C'][0], random.uniform(0, 2*math.pi))
            #mb.rotate_around_y(self['C'][0], random.uniform(0, 2 * math.pi))
            #mb.rotate_around_z(self['C'][0], random.uniform(0, 2 * math.pi))

        for index in range(num_particles - 1):
            mb.force_overlap(move_this=self['C'][index + 1],
                             from_positions=self['C'][index + 1]['down'],
                             to_positions=self['C'][index]['up'])
