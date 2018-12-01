from epoxpy.neat_toughener_simulation import NeatToughenerSimulation
from epoxpy.lib import A, B, C, C10, CXX, Epoxy_A_10_B_20_C10_2_Blend
import epoxpy.init as my_init
import hoomd
from hoomd import md
from hoomd import deprecated
from hoomd import variant
import mbuild as mb
import epoxpy.common as cmn
from epoxpy.utils import Angles
import numpy as np


class NeatToughenerLJHarmonicSimulation(NeatToughenerSimulation):
    """Simulations class for NeatToughenerSimulation where LJ is used as the
    conservative force and uses the langevin integrator.
       """

    def __init__(self,
                 sim_name,
                 mix_time,
                 mix_kt,
                 temp_prof,
                 CC_interaction=1.0,
                 CC_sigma=1.0,
                 shrink_time=1e6,
                 shrinkT=2.0,
                 tau=0.1,
                 tauP=0.2,
                 P=1.0,
                 integrator=cmn.Integrators.LANGEVIN.name,
                 *args,
                 **kwargs):
        NeatToughenerSimulation.__init__(self,
                                        sim_name,
                                        mix_time,
                                        mix_kt,
                                        temp_prof,
                                        *args,
                                        **kwargs)
        self.CC_interaction = CC_interaction
        self.CC_sigma = CC_sigma
        self.shrink_time = shrink_time
        self.shrinkT = shrinkT
        self.integrator = cmn.Integrators[integrator]
        self.tau = tau
        self.tauP = tauP
        self.P = P
        self._exclude_bonds_from_nlist = True
        self.T_integrator = None
        self.cure_kt = None

    def get_log_quantities(self):
        log_quantities = super().get_log_quantities()+["pair_lj_energy", "bond_harmonic_energy"]
        return log_quantities

    def get_non_bonded_neighbourlist(self, nl_type="tree"):
        if nl_type == "tree":
            nl = md.nlist.tree()  # cell()
        elif nl_type == "cell":
            nl = md.nlist.cell()
        else:
            raise NotImplementedError

        nl.reset_exclusions(exclusions=['bond'])

        return nl

    def set_initial_structure(self):
        print('========INITIAIZING FOR LJ==========')
        desired_box_volume = ((C.mass*self.num_cXX*self.cXX)) / self.density
        desired_box_dim = (desired_box_volume ** (1./3.))
        reduced_density = self.density / 10
        ex_box_vol = (C.mass*self.num_cXX*self.cXX) / reduced_density
        expanded_box_dim = (ex_box_vol ** (1. / 3.))
        half_L = expanded_box_dim / 2
        box = mb.Box(mins=[-half_L, -half_L, -half_L], maxs=[half_L, half_L, half_L])
        if self.old_init:
            raise NotImplementedError('Old init method not implemented or this simulation')
        else:
            print('Packing {} C10 particles ..'.format(self.num_cXX))
            mix_box = mb.packing.fill_box([CXX(numCs=self.cXX)],
                                          [self.num_cXX],
                                          box=box,
                                          seed=self.mbuild_seed)

            if self.init_file_name.endswith('.hoomdxml'):
                mix_box.save(self.init_file_name, overwrite=True, ref_distance=.1)
            elif self.init_file_name.endswith('.gsd'):
                mix_box.save(self.init_file_name, write_ff=False, overwrite=True)

            if self.init_file_name.endswith('.hoomdxml'):
                self.system = hoomd.deprecated.init.read_xml(self.init_file_name, wrap_coordinates=True)
            elif self.init_file_name.endswith('.gsd'):
                self.system = hoomd.init.read_gsd(self.init_file_name)

            print('Initial box dimension: {}'.format(self.system.box.dimensions))

            snapshot = self.system.take_snapshot(bonds=True)
            for p_id in range(snapshot.particles.N):
                p_types = snapshot.particles.types
                p_type = p_types[snapshot.particles.typeid[p_id]]
                if p_type == 'C':
                    snapshot.particles.mass[p_id] = C.mass
                else:
                    raise ValueError('Expecting only C type particles. Recieved {}'.format(p_type))
            print(snapshot.bonds.types)
            snapshot.bonds.types = ['C-C']
            angles = Angles()
            ccc_angles = angles.get_angles_for_linear_chains(snapshot, 'C', simple=True, lin_chain_N=10)
            snapshot.angles.types = ['C-C-C']
            for ccc_angle in ccc_angles:
                n_angles = snapshot.angles.N
                snapshot.angles.resize(n_angles + 1)
                snapshot.angles.group[n_angles] = ccc_angle
                snapshot.angles.typeid[n_angles] = 0  # we know C-C-C bond angle's id is 1
            self.system.restore_snapshot(snapshot)

        if self.shrink is True:
            self.nl = self.get_non_bonded_neighbourlist()
            self.setup_force_fields(stage=cmn.Stages.SHRINKING)
            size_variant = variant.linear_interp([(0, self.system.box.Lx), (self.shrink_time, desired_box_dim)])
            md.integrate.mode_standard(dt=self.mix_dt)
            shrink_integrator = md.integrate.langevin(group=hoomd.group.all(),
                                  kT=self.shrinkT,
                                  seed=1223445)  # self.seed)
            shrink_integrator.set_gamma('C', gamma=self.gamma)

            resize = hoomd.update.box_resize(L=size_variant)
            hoomd.run(self.shrink_time)
            snapshot = self.system.take_snapshot()
            print('Initial box dimension: {}'.format(snapshot.box))

        if self.init_file_name.endswith('.hoomdxml'):
            deprecated.dump.xml(group=hoomd.group.all(), filename=self.init_file_name, all=True)
        elif self.init_file_name.endswith('.gsd'):
            hoomd.dump.gsd(group=hoomd.group.all(), filename=self.init_file_name, overwrite=True, period=None)
        return self.system

    def setup_force_fields(self, stage):
        if self.DEBUG:
            print('=============force fields parameters==============')
            print('self.CC_bond_const', self.CC_bond_const)
            print('self.CC_bond_dist', self.CC_bond_dist)
            print('self.CC_bond_angle_const', self.CC_bond_angle_const)
            print('self.CC_bond_angle', self.CC_bond_angle)
            print('self.gamma', self.gamma)
        harmonic = md.bond.harmonic()
        if self.num_cXX > 0:
            harmonic.bond_coeff.set('C-C', k=self.CC_bond_const, r0=self.CC_bond_dist)

        if self.CC_bond_angle_const is not None and self.CC_bond_angle is not None:
            angle = md.angle.cosinesq()
            angle.angle_coeff.set('C-C-C', k=self.CC_bond_angle_const, t0=np.radians(self.CC_bond_angle))
            print('################### Setting angle potential for C-C-C .')

        if stage == cmn.Stages.SHRINKING:
            self.nl = self.get_non_bonded_neighbourlist(nl_type="tree")
        else:
            self.nl = self.get_non_bonded_neighbourlist(nl_type="cell")

        lj = md.pair.lj(r_cut=2.5, nlist=self.nl)
        lj.pair_coeff.set('C', 'C', epsilon=self.CC_interaction, sigma=self.CC_sigma)
        lj.set_params(mode="xplor")

    def setup_integrator(self, stage):
        print('=============Setting up {} integrator for {}'.format(self.integrator.name, stage.name))
        if stage == cmn.Stages.MIXING:
            temperature = self.mix_kT
            dt = self.mix_dt
            print('========= MIXING TEMPERATURE:', temperature, '=============')
        elif stage == cmn.Stages.CURING:
            temperature = self.temp_prof.get_profile()
            if self.enable_rxn_enthalpy:
                all_temperatures = np.asarray(self.temp_prof.temperature_profile)[:, 1]
                if all_temperatures.std() == 0:
                    self.cure_kt = all_temperatures[0]
                else:
                    raise NotImplementedError('Enthalpy change for non isothermal cure is not implemented')
            dt = self.md_dt
            print('========= CURING TEMPERATURE:', temperature, '=============')
        md.integrate.mode_standard(dt=dt)
        if self.integrator == cmn.Integrators.LANGEVIN:
            self.T_integrator = md.integrate.langevin(group=hoomd.group.all(),
                                               kT=temperature,
                                               seed=1223445,
                                               noiseless_t=False,
                                               noiseless_r=False)
            self.T_integrator.set_gamma('C', gamma=self.gamma)
        elif self.integrator == cmn.Integrators.NPT:
            self.T_integrator = md.integrate.npt(group=hoomd.group.all(),
                                          tau=self.tau,
                                          tauP=self.tauP,
                                          P=self.P,
                                          kT=temperature)
        elif self.integrator == cmn.Integrators.NVT:
            self.T_integrator = md.integrate.nvt(group=hoomd.group.all(),
                                          tau=self.tau,
                                          kT=temperature)

    def reset_setpoint_temperature(self, timestep, deltaT):
        # current_T = self.cure_kt
        new_T = self.cure_kt + deltaT
        # print('changing set point from {} to {}, deltaT:{}'.format(current_T, new_T, deltaT))
        self.T_integrator.set_params(kT=new_T)
        self.cure_kt = new_T
