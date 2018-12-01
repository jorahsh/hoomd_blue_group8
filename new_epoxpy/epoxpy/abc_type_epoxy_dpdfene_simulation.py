from epoxpy.abc_type_epoxy_simulation import ABCTypeEpoxySimulation
import hoomd
from hoomd import md
import epoxpy.common as cmn


class ABCTypeEpoxyDPDFENESimulation(ABCTypeEpoxySimulation):
    """Simulations class for ABCTypeEpoxySimulation where LJ is used as the
    conservative force in the DPD force field.
       """
    def __init__(self,
                 sim_name,
                 mix_time,
                 mix_kt,
                 temp_prof,
                 AA_interaction=1.0,
                 AB_interaction=1.0,
                 AC_interaction=1.0,
                 BC_interaction=1.0,
                 AA_alpha=0.0,
                 AB_alpha=0.0,
                 AC_alpha=0.0,
                 BC_alpha=0.0,
                 *args,
                 **kwargs):
        ABCTypeEpoxySimulation.__init__(self,
                                        sim_name,
                                        mix_time,
                                        mix_kt,
                                        temp_prof,
                                        *args,
                                        **kwargs)
        self.AA_interaction = AA_interaction
        self.AB_interaction = AB_interaction
        self.AC_interaction = AC_interaction
        self.BC_interaction = BC_interaction
        self.AA_alpha = AA_alpha
        self.AB_alpha = AB_alpha
        self.AC_alpha = AC_alpha
        self.BC_alpha = BC_alpha
        self.CC_maxr = 1.5
        self.AB_maxr = 1.5
        self._exclude_bonds_from_nlist = False

    def exclude_bonds_from_nlist(self):
        return self._exclude_bonds_from_nlist

    def get_log_quantities(self):
        log_quantities = super().get_log_quantities()+["pair_dpdlj_energy","bond_fene_energy"]
        return log_quantities

    def get_non_bonded_neighbourlist(self):
        nl = md.nlist.tree()  # cell()
        nl.reset_exclusions(exclusions=[]);
        return nl

    def setup_force_fields(self, stage):
        if self.DEBUG:
            print('=============force fields parameters==============')
            print('self.CC_bond_const', self.CC_bond_const)
            print('self.CC_bond_dist', self.CC_bond_dist)
            print('self.AB_bond_const', self.AB_bond_const)
            print('self.AB_bond_dist', self.AB_bond_dist)
            print('self.AA_interaction', self.AA_interaction)
            print('self.AB_interaction', self.AB_interaction)
            print('self.AC_interaction', self.AC_interaction)
            print('self.BC_interaction', self.BC_interaction)
            print('self.gamma', self.gamma)

        if stage == cmn.Stages.MIXING:
            temperature = self.mix_kT
            print('========= MIXING TEMPERATURE:', temperature, '=============')
        elif stage == cmn.Stages.CURING:
            profile = self.temp_prof.get_profile()
            temperature = profile
            print('========= CURING TEMPERATURE:', temperature, '=============')

        fene = md.bond.fene()
        if self.num_b > 0 and self.num_a > 0:
            fene.bond_coeff.set('A-B', k=self.AB_bond_const,
                                r0=self.AB_maxr, sigma=self.AB_bond_dist, epsilon=self.AB_interaction)
        if self.num_c10 > 0:
            fene.bond_coeff.set('C-C', k=self.CC_bond_const,
                                r0=self.CC_maxr, sigma=self.CC_bond_dist, epsilon=1.0)

        dpdlj = md.pair.dpdlj(r_cut=2.5, nlist=self.nl, kT=temperature, seed=123456)
        dpdlj.pair_coeff.set('A', 'A', epsilon=self.AA_interaction, sigma=1.0, gamma=self.gamma, alpha=self.AA_alpha)
        dpdlj.pair_coeff.set('B', 'B', epsilon=self.AA_interaction, sigma=1.0, gamma=self.gamma, alpha=self.AA_alpha)
        dpdlj.pair_coeff.set('C', 'C', epsilon=self.AA_interaction, sigma=1.0, gamma=self.gamma, alpha=self.AA_alpha)

        dpdlj.pair_coeff.set('A', 'B', epsilon=self.AB_interaction, sigma=1.0, gamma=self.gamma, alpha=self.AB_alpha)
        dpdlj.pair_coeff.set('A', 'C', epsilon=self.AC_interaction, sigma=1.0, gamma=self.gamma, alpha=self.AC_alpha)
        dpdlj.pair_coeff.set('B', 'C', epsilon=self.BC_interaction, sigma=1.0, gamma=self.gamma, alpha=self.BC_alpha)

    def setup_integrator(self, stage):
        if stage == cmn.Stages.MIXING:
            dt = self.mix_dt
            print('========= MIXING dt:', dt, '=============')
        elif stage == cmn.Stages.CURING:
            dt = self.md_dt
            print('========= CURING dt:', dt, '=============')
        md.integrate.mode_standard(dt=dt)
        md.integrate.nve(group=hoomd.group.all())

    def reset_setpoint_temperature(self, timestep, deltaT):
        raise NotImplementedError('reset_setpoint_temperature not implemented for', self)
