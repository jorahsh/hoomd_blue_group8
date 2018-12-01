from __future__ import division
import numpy as np
import random
from hoomd import * # I really don't like global imports like this

class Bead:

    def __init__(self, diameter=1.0, mass=1.0, charge=0.0, btype="A", N=1):
        self.N = N
        self.diameter = diameter
        self.mass = mass
        self.charge = charge
        self.btype = btype

class PolyBead(Bead):

    def __init__(self, diameter=1.0, mass=1.0, charge=0.0, btype="C", N=5):
        Bead.__init__(self, diameter, mass, charge, btype, N)
        self.bond_type = "{}-{}".format(self.btype, self.btype)
        self.bond_list = np.array([_ for _ in range(N)])



def init_system(building_block_dic, rho):
    """ building_block_dic should be a dctionay of beads/polybeads : # """
    context.initialize("")
    N = 0
    system_mass = 0
    particle_types = []
    bond_types = []
    typeid_list = []
    mass_list = []
    diameter_list = []
    point_list = []
    bond_list = []
    i = 0
    offset = 0
    for bead, N_b in building_block_dic.items():
        N += N_b * bead.N
        particle_types.append(bead.btype)
        system_mass += bead.mass * N_b * bead.N
        mass_list += [bead.mass] * N_b * bead.N
        diameter_list += [bead.diameter] * N_b* bead.N
        typeid_list += [i] * N_b * bead.N
        if bead.N > 1:
            bond_types.append(bead.bond_type)
            for k in range(N_b):
                bnd_list_str = bead.bond_list +  k*bead.N + offset
                for a,b in zip(bnd_list_str, bnd_list_str[1:]):
                    #print("{}-{}".format(a, b))
                    bond_list.append("{}-{}".format(a, b))
        offset += N_b * bead.N
        #print("offset",offset)
        i += 1
    L = (system_mass/rho)**(1.0/3.0)
    # Hard code in the A-B bond type
    bond_types.append("A-B")
    snap = data.make_snapshot(N=N,
            box=data.boxdim(L=L),
            particle_types=particle_types,
            bond_types=bond_types)
    for _ in range(N):
        x = random.uniform(-L/2.0, L/2.0)
        y = random.uniform(-L/2.0, L/2.0)
        z = random.uniform(-L/2.0, L/2.0)
        point_list.append([x, y, z])
    snap.particles.typeid[:] = typeid_list
    snap.particles.mass[:] = mass_list
    snap.particles.position[:] = point_list
    for bond in bond_list:
        #print(bond)
        #print(bond.split("-")[0])
        #print(bond.split("-")[1])
        bond_a = int(bond.split("-")[0])
        bond_b = int(bond.split("-")[1])
        n_bonds = snap.bonds.N
        snap.bonds.resize(n_bonds + 1)
        snap.bonds.group[n_bonds] = [bond_a, bond_b]
        snap.bonds.typeid[n_bonds] = 0
    #print(bond_list)
    return snap
