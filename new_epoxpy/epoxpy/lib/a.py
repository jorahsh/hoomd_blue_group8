import mbuild as mb


class A(mb.Compound):
    """An coarse grained particle called A(arbitrarily) with two ports one facing up and one facing down."""
    mass = 1.0

    def __init__(self):
        super(A, self).__init__()

        self.add(mb.Particle(name='A'))

        self.add(mb.Port(anchor=self[0]), 'up')
        mb.Compound.translate(self['up'], [0, 0.07, 0])

        self.add(mb.Port(anchor=self[0]), 'down')
        mb.Compound.translate(self['down'], [0, -0.07, 0])
