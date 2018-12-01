import mbuild as mb


class B(mb.Compound):
    """An coarse grained particle called B(arbitrarily) with two ports one facing up and one facing down."""
    mass = 1.0

    def __init__(self):
        super(B, self).__init__()

        self.add(mb.Particle(name='B'))

        self.add(mb.Port(anchor=self[0]), 'up')
        mb.Compound.translate(self['up'], [0, 0.07, 0])

        self.add(mb.Port(anchor=self[0]), 'down')
        mb.Compound.translate(self['down'], [0, -0.07, 0])
