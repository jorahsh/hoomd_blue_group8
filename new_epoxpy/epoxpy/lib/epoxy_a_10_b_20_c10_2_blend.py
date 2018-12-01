import random
import mbuild as mb
from epoxpy.lib import A, B, C10


class Epoxy_A_10_B_20_C10_2_Blend(mb.Compound):
    """A blend of coarse grained molecules A, B and C10 where the ratio of A:B:C10 is 10:20:2 .

        Note:
            Positions the A's, B's and C10's are all at origin.

        Parameters
        ----------
        box : np.ndarray, shape=(3,), dtype=float
        The dimensions of the box containing the blend

        """

    def __init__(self, box=[1, 1, 1]):
        super(Epoxy_A_10_B_20_C10_2_Blend, self).__init__()

        if len(box) is not 3:
            raise ValueError('box shape should be 3. You passed in {}'.format(len(box)))

        num_a = 10
        num_b = 20
        num_c10 = 2

        for index in range(0, num_a):
            self.add(A(), label='A[$]')
            x = random.uniform(-box[0] / 2, box[0] / 2)
            y = random.uniform(-box[1] / 2, box[1] / 2)
            z = random.uniform(-box[2] / 2, box[2] / 2)
            mb.Compound.translate(self['A'][-1], [x, y, z])

        for index in range(0, num_b):
            self.add(B(), label='B[$]')
            x = random.uniform(-box[0] / 2, box[0] / 2)
            y = random.uniform(-box[1] / 2, box[1] / 2)
            z = random.uniform(-box[2] / 2, box[2] / 2)
            mb.Compound.translate(self['B'][-1], [x, y, z])

        for index in range(0, num_c10):
            self.add(C10(), label='C10[$]')
            x = random.uniform(-box[0] / 2, box[0] / 2)
            y = random.uniform(-box[1] / 2, box[1] / 2)
            z = random.uniform(-box[2] / 2, box[2] / 2)
            mb.Compound.translate(self['C10'][-1], [x, y, z])
