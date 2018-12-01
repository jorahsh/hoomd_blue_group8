import hoomd
import hoomd.dybond_plugin as db;
import hoomd.md.pair as pair
import numpy as np
import hoomd.deprecated
import random


def printandstopdybond(timestep, cure_percent, delta_T,current_set_T, dpd):
    print('############### Cured ', cure_percent, 'temperature', delta_T, current_set_T[0], '#################')
    current_set_T[0] = current_set_T[0] + delta_T
    dpd.set_params(kT=current_set_T[0])
    #updater1.disable()

#initialize hoomd
hoomd.context.initialize();
random.seed(1020)
unitcell=hoomd.lattice.sc(a=1.1, type_name='A')
system = hoomd.init.create_lattice(unitcell=unitcell, n=7)
snapshot = system.take_snapshot(bonds=True)
snapshot.particles.types = ['A', 'B']
N = snapshot.particles.N
typeIds = np.zeros(N)
typeIds[int(N/3):]=1
#typeIds[::2] = 1
snapshot.particles.typeid[:] =typeIds
print(snapshot.particles.typeid)
snapshot.bonds.types = ['A-B']
snapshot.bonds.resize(1)
# need to create atleast one bond for defining the harmonic potential
snapshot.bonds.group[0] = [0,N-1]
snapshot.bonds.typeid[0] = 0
#my_velocity = np.random.random((N,3)) * 2 - 1;
#snapshot.particles.velocity[:] = my_velocity[:];

print(snapshot.bonds.types,snapshot.bonds.typeid,N)
system.restore_snapshot(snapshot)
#create a neighbor list that is being reused for both pair force computation and bonding

nl = hoomd.md.nlist.cell()
AA_interaction = 1.0
AC_interaction = 10.0

dpd = hoomd.md.pair.dpd(r_cut=2.5, nlist=nl, kT=200.0, seed=123450)
dpd.pair_coeff.set('A', 'A', A=AA_interaction, gamma=1.0)
dpd.pair_coeff.set('B', 'B', A=AA_interaction, gamma=1.0)
dpd.pair_coeff.set('A', 'B', A=AC_interaction, gamma=1.0)

harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('A-B', k=100.0, r0=1.0)

all = hoomd.group.all();
hoomd.md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.nve(group=all)

hoomd.dump.gsd(filename='data.gsd',
               period=100,
               group=hoomd.group.all(),
               static=['attribute'],
               overwrite=True)
hoomd.run(1e1)
current_set_T = [1.5]
dpd.set_params(kT = current_set_T[0])
#setup the dynamic bond updater and reuse the neighbour list
updater = db.update.dybond(nl, group=all, period=10)
updater.set_params(bond_type='A-B',
                   A='A',
                   A_fun_groups=4,
                   B='B',
                   B_fun_groups=2,
                   rcut=3.0,
                   Ea=1.0,
                   alpha=2.0,
                   percent_bonds_per_step=0.5,
                   stop_after_percent=75.0,
                   enable_rxn_enthalpy=True,#False,
                   deltaT=0.001,
                   callback=lambda timestep, b_percent, delta_T :
                   printandstopdybond(timestep, b_percent, delta_T, current_set_T, dpd))
#updater.set_params(bond_type='A-C',A='A',A_fun_groups=4,B='C',B_fun_groups=2,rcut=1.0,Ea=1.0,alpha=2.0)
hoomd.analyze.log(filename='out.log',
                  quantities=["pair_dpd_energy",
                              "volume",
                              "momentum",
                              "potential_energy",
                              "kinetic_energy",
                              "temperature",
                              "pressure",
                              "bond_harmonic_energy",
                              "bond_percent(A-B)",
                              "bonds_per_step(A-B)"],
                  period=1,
                  header_prefix='#',
                  overwrite=True)
hoomd.dump.dcd(filename='traj.dcd',
               period=1e3,
               overwrite=True)
hoomd.run(1e4,profile=False)
print("Simulation done..")
hoomd.deprecated.dump.xml(group=hoomd.group.all(), filename='final.hoomdxml', all=True)
del system
