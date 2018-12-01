from epoxpy.a_type_epoxy_simulation import ATypeEpoxySimulation
from epoxpy.lib import A
import epoxpy.init as my_init
import hoomd
from hoomd import md
from hoomd import deprecated
from hoomd import variant
import mbuild as mb
import epoxpy.common as cmn


class ATypeEpoxyLJHarmonicSimulation(ATypeEpoxySimulation):
    """Simulations class for ABCTypeEpoxySimulation where LJ is used as the
    conservative force and uses the langevin integrator.
       """
    def __init__(self,
                 sim_name,
                 mix_time,
                 mix_kt,
                 temp_prof,
                 AA_interaction=1.0,
                 shrink_time=1e6,
                 shrinkT=2.0,
                 AA_alpha=1.0,
                 tau=0.1,
                 tauP=0.2,
                 P=1.0,
                 integrator=cmn.Integrators.LANGEVIN.name,
                 *args,
                 **kwargs):
        ATypeEpoxySimulation.__init__(self,
                                        sim_name,
                                        mix_time,
                                        mix_kt,
                                        temp_prof,
                                        *args,
                                        **kwargs)
        self.AA_interaction = AA_interaction

        self.AA_alpha = AA_alpha
        self.shrink_time = shrink_time
        self.shrinkT = shrinkT
        self.integrator = cmn.Integrators[integrator]
        self.tau = tau
        self.tauP = tauP
        self.P = P
        self._exclude_bonds_from_nlist = True

    def exclude_bonds_from_nlist(self):
        return self._exclude_bonds_from_nlist

    def get_log_quantities(self):
        if self.bond:
            log_quantities = super().get_log_quantities()+["pair_lj_energy", "bond_harmonic_energy"]
        else:
            log_quantities = super().get_log_quantities() + ["pair_lj_energy"]
        return log_quantities

    def get_non_bonded_neighbourlist(self):
        nl = md.nlist.cell()
        nl.reset_exclusions(exclusions=['bond']);
        return nl

    def set_initial_structure(self):
        print('========INITIAIZING FOR LJ==========')
        desired_box_volume = (A.mass*self.num_a) / self.density
        desired_box_dim = (desired_box_volume ** (1./3.))
        half_L = desired_box_dim / 2
        box = mb.Box(mins=[-half_L, -half_L, -half_L], maxs=[half_L, half_L, half_L])
        if self.old_init == True:
            print("\n\n ===USING OLD INIT=== \n\n")
            As = my_init.Bead(btype="A", mass=A.mass)
            snap = my_init.init_system({As : int(self.num_a)}, self.density/10)
            self.system = hoomd.init.read_snapshot(snap)
        else:
            if self.shrink is True:
                print('Packing {} A particles ..'.format(self.num_a))
                mix_box = mb.packing.fill_box(A(), self.num_a,
                                              box=box)#,overlap=0.5)

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
                if p_type == 'A':
                    snapshot.particles.mass[p_id] = A.mass
                else:
                    raise ValueError('Not expecting particles other than A here!')
            print(snapshot.bonds.types)
            snapshot.bonds.types = ['A-A']
            self.system.restore_snapshot(snapshot)

        if self.shrink is True:
            self.nl = self.get_non_bonded_neighbourlist()
            if self.bond:
                self.make_one_bond()
            self.setup_force_fields(stage=cmn.Stages.MIXING)
            size_variant = variant.linear_interp([(0, self.system.box.Lx), (self.shrink_time, desired_box_dim)])
            md.integrate.mode_standard(dt=self.mix_dt)
            md.integrate.langevin(group=hoomd.group.all(),
                                  kT=self.shrinkT,
                                  seed=1223445)  # self.seed)
            resize = hoomd.update.box_resize(L=size_variant)
            hoomd.run(self.shrink_time)
            snapshot = self.system.take_snapshot()
            print('Initial box dimension: {}'.format(snapshot.box))

        if self.init_file_name.endswith('.hoomdxml'):
            deprecated.dump.xml(group=hoomd.group.all(), filename=self.init_file_name, all=True)
        elif self.init_file_name.endswith('.gsd'):
            hoomd.dump.gsd(group=hoomd.group.all(), filename=self.init_file_name, overwrite=True, period=None)

        return self.system

    def make_one_bond(self):
        if self.system is not None:
            snapshot = self.system.take_snapshot(all=True)
            n_bonds = snapshot.bonds.N
            snapshot.bonds.resize(n_bonds + 1)
            if snapshot.particles.N < 2:
                raise ValueError('We need alteast two particles to make bonds!')
            snapshot.bonds.group[n_bonds] = [0, 1]#bond particle 1 and 2 (assuming there are atleast two particles here
            # sets new bond to be A-A type
            snapshot.bonds.typeid[n_bonds] = 0
            self.system.restore_snapshot(snapshot)
        else:
            raise ValueError('bonds being added before initializing system')

    def setup_force_fields(self, stage):
        if self.DEBUG:
            print('=============force fields parameters==============')
            print('self.AA_bond_const', self.AA_bond_const)
            print('self.AA_interaction', self.AA_interaction)
            print('self.gamma', self.gamma)

        if self.bond:
            harmonic = md.bond.harmonic()
            harmonic.bond_coeff.set('A-A', k=self.AA_bond_const, r0=self.AA_bond_dist)

        lj = md.pair.lj(r_cut=2.5, nlist=self.nl)
        lj.pair_coeff.set('A', 'A', epsilon=self.AA_interaction, sigma=1.0, alpha=self.AA_alpha)
        lj.set_params(mode="xplor")

    def setup_integrator(self, stage):
        print('=============Setting up {} integrator for {}'.format(self.integrator.name, stage.name))
        if stage == cmn.Stages.MIXING:
            temperature = self.mix_kT
            dt = self.mix_dt
            print('========= MIXING TEMPERATURE:', temperature, '=============')
        elif stage == cmn.Stages.CURING:
            profile = self.temp_prof.get_profile()
            temperature = profile
            dt = self.md_dt
            print('========= CURING TEMPERATURE:', temperature, '=============')
        md.integrate.mode_standard(dt=dt)
        if self.integrator == cmn.Integrators.LANGEVIN:
            integrator = md.integrate.langevin(group=hoomd.group.all(),
                                               kT=temperature,
                                               seed=1223445,
                                               noiseless_t=False,
                                               noiseless_r=False)
            integrator.set_gamma('A', gamma=self.gamma)
        elif self.integrator == cmn.Integrators.NPT:
            integrator = md.integrate.npt(group=hoomd.group.all(),
                                          tau=self.tau,
                                          tauP=self.tauP,
                                          P=self.P,
                                          kT=temperature)
        elif self.integrator == cmn.Integrators.NVT:
            integrator = md.integrate.nvt(group=hoomd.group.all(),
                                          tau=self.tau,
                                          kT=temperature)

    def reset_setpoint_temperature(self, timestep, deltaT):
        raise NotImplementedError('reset_setpoint_temperature not implemented for', self)
