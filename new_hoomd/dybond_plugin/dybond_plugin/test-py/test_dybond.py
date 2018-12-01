# -*- coding: iso-8859-1 -*-
# Maintainer: sthomas

import hoomd;
import hoomd.dybond_plugin as db;
import unittest;
import os;
import numpy as np

def call_back(timestep, cure_percent, delta_T):
    print('############### Cured ', cure_percent, 'temperature', delta_T, '#################')

class test_simple(unittest.TestCase):
    def test_small_run(self):
        hoomd.context.initialize();
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                           n=[30,30]);
        snapshot = system.take_snapshot(bonds=True)
        snapshot.particles.types = ['A', 'B']
        N = snapshot.particles.N
        typeIds = np.zeros(N)
        typeIds[int(N/2):]=1
        snapshot.particles.typeid[:] =typeIds
        print(snapshot.particles.typeid)
        snapshot.bonds.types = ['A-B']
        snapshot.bonds.resize(1)
        # need to create atleast one bond for defining the harmonic potential
        snapshot.bonds.group[0] = [0,N-1]
        snapshot.bonds.typeid[0] = 0

        print(snapshot.bonds.types,snapshot.bonds.typeid,N)
        system.restore_snapshot(snapshot)

        nl = hoomd.md.nlist.cell()
        updater = db.update.dybond(nl, group=hoomd.group.all(), period=10)
        updater.set_params(bond_type='A-B',
                           A='A',
                           A_fun_groups=4,
                           B='B',
                           B_fun_groups=2,
                           rcut=1.0,
                           Ea=1.0,
                           alpha=2.0,
                           percent_bonds_per_step=0.0025,
                           stop_after_percent=100., 
                           exclude_from_nlist=True,
                           enable_rxn_enthalpy=True,
                           deltaT=0.001,
                           callback=lambda timestep, b_percent, delta_T :call_back(timestep, b_percent, delta_T))
        hoomd.dump.gsd(filename='data.gsd',period=1, group=hoomd.group.all(), overwrite=True)
        hoomd.analyze.log(filename='out.log', quantities=["pair_dpd_energy", "volume",
                                                  "momentum",
                                                  "potential_energy",
                                                  "kinetic_energy","temperature",
                                                  "pressure",
                                                  "bond_harmonic_energy","bond_percent(A-B)"], period=1e3,header_prefix='#', overwrite=True)

        hoomd.run(240000)
        del system
if __name__ == '__main__':
    unittest.main(argv = ['test_dybond.py', '-v'])
