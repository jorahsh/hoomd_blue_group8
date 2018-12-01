import epoxpy.abc_type_epoxy_lj_harmonic_simulation as es
import epoxpy.temperature_profile_builder as tpb
import epoxpy.common as cmn
import random
import os

random.seed(1020)

mix_time = 1e2
mix_kt = 2.0
cure_kt = 2.0
time_scale = 1
temp_scale = 1
type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt, initial_time=mix_time)
type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)

out_dir = str('.')
sim_name = 'langH'
out_dir = os.path.join(out_dir, sim_name)
myEpoxySim = es.ABCTypeEpoxyLJHarmonicSimulation(sim_name,
                                                 mix_time=mix_time,
                                                 mix_kt=mix_kt,
                                                 temp_prof=type_A_md_temp_profile,
                                                 output_dir=out_dir,
                                                 CC_bond_angle=109.5,
                                                 CC_bond_angle_const=100,
                                                 bond=True, n_mul=15.0, shrink=True,
                                                 shrink_time=1e5,
                                                 mix_dt=1e-4,
                                                 md_dt=1e-2,
                                                 integrator=cmn.Integrators.LANGEVIN.name,
                                                 use_dybond_plugin=True)

myEpoxySim.execute()
