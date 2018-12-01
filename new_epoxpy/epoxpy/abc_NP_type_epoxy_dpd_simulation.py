from epoxpy.abc_type_epoxy_dpd_simulation import ABCTypeEpoxyDPDSimulation
import hoomd
from hoomd import md
import epoxpy.common as cmn
from hoomd import variant
import mbuild as mb
import epoxpy.init as my_init
from epoxpy.lib import A, B, C, Sphere


class ABCNPTypeEpoxyDPDSimulation(ABCTypeEpoxyDPDSimulation):
    """Simulations class for ABCTypeEpoxySimulation where nano particles are used as
    toughener instead of toughener chains and where DPD is used as the
    conservative force.
       """
    def __init__(self,
                 sim_name,
                 mix_time,
                 mix_kt,
                 temp_prof,
                 AA_interaction=25.0,
                 AB_interaction=35.0,
                 AC_interaction=35.0,
                 BC_interaction=35.0,
                 num_spheres=2,
                 nlist=cmn.NeighbourList.CELL.name,
                 *args,
                 **kwargs):
        ABCTypeEpoxyDPDSimulation.__init__(self,
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
        self.nlist = cmn.NeighbourList[nlist]
        self.num_spheres = num_spheres
        self.dpd = None
        self._exclude_bonds_from_nlist = False

    def set_initial_structure(self):
        print('========INITIAIZING STRUCTURE==========')
        num_beads_in_sphere = 65
        Sphere.mass = num_beads_in_sphere
        Sphere.name = 'Sphere'
        desired_box_volume = ((A.mass*self.num_a) + (B.mass*self.num_b) + (Sphere.mass*self.num_spheres)) / self.density
        desired_box_dim = (desired_box_volume ** (1./3.))
        reduced_density = self.density/10
        ex_box_vol = ((A.mass * self.num_a) + (B.mass * self.num_b) + (Sphere.mass*self.num_spheres)) / reduced_density
        expanded_box_dim = (ex_box_vol ** (1. / 3.))
        half_L = expanded_box_dim/2
        box = mb.Box(mins=[-half_L, -half_L, -half_L], maxs=[half_L, half_L, half_L])
        if self.old_init:
            raise NotImplementedError('Spherical Nano Particles not implemented in old init method')
        else:
            print("\n\n ===USING MBUILD INIT=== \n\n")
            if self.shrink is False:
                print('shrink=False is deprecated.')
            print('Packing {} A particles, {} B particles and {} C65s ..'.format(self.num_a,
                                                                                 self.num_b,
                                                                                 self.num_spheres))
            
            
            mix_box = mb.packing.fill_box([A(), B(), Sphere(n=num_beads_in_sphere)],
                                          [self.num_a, self.num_b, self.num_spheres],
                                          box=box)  # ,overlap=0.5)
            if self.num_spheres > 0: 
                mix_box.label_rigid_bodies(discrete_bodies = 'Sphere')

            if self.init_file_name.endswith('.hoomdxml'):
                mix_box.save(self.init_file_name, overwrite=True, ref_distance=.1)
            elif self.init_file_name.endswith('.gsd'):
                mix_box.save(self.init_file_name, write_ff=False, overwrite=True)

            if self.init_file_name.endswith('.hoomdxml'):
                system = hoomd.deprecated.init.read_xml(self.init_file_name, wrap_coordinates=True)
            elif self.init_file_name.endswith('.gsd'):
                system = hoomd.init.read_gsd(self.init_file_name)

            print('Initial box dimension: {}'.format(self.system.box.dimensions))
            return system

    def reset_setpoint_temperature(self, timestep, deltaT):
        raise NotImplementedError('reset_setpoint_temperature not implemented for', self)

#            snapshot = self.system.take_snapshot(bonds=True)
#            for p_id in range(snapshot.particles.N):
#                p_types = snapshot.particles.types
#                p_type = p_types[snapshot.particles.typeid[p_id]]
#                if p_type == 'A':
#                    snapshot.particles.mass[p_id] = A.mass
#                if p_type == 'B':
#                    snapshot.particles.mass[p_id] = B.mass
#                if p_type == 'C':
#                    snapshot.particles.mass[p_id] = C.mass
#            print(snapshot.bonds.types)
#            snapshot.bonds.types = ['C-C', 'A-B']
#            self.system.restore_snapshot(snapshot)
#
#        self.nl = self.get_non_bonded_neighbourlist()
#        self.setup_force_fields(stage=cmn.Stages.MIXING)
#        size_variant = variant.linear_interp([(0, self.system.box.Lx), (self.shrink_time, desired_box_dim)])
#        md.integrate.mode_standard(dt=self.mix_dt)
#        md.integrate.langevin(group=hoomd.group.all(),
#                              kT=self.shrinkT,
#                              seed=1223445)  # self.seed)
#        resize = hoomd.update.box_resize(L=size_variant)
#        hoomd.run(self.shrink_time)
#        snapshot = self.system.take_snapshot()
#        print('Initial box dimension: {}'.format(snapshot.box))

#        if self.init_file_name.endswith('.hoomdxml'):
#            deprecated.dump.xml(group=hoomd.group.all(), filename=self.init_file_name, all=True)
#        elif self.init_file_name.endswith('.gsd'):
#            hoomd.dump.gsd(group=hoomd.group.all(), filename=self.init_file_name, overwrite=True, period=None)
