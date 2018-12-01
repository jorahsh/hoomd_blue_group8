import epoxpy.abc_NP_type_epoxy_lj_harmonic_simulation as es
import epoxpy.temperature_profile_builder as tpb
import epoxpy.bonding as bondClass
import random
import os
import gsd.hoomd
import numpy as np

random.seed(1020)

mix_time = 1e4
mix_kt = 2.0
cure_kt = 2.0
cure_time = 3e4
temp_scale = 1
type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt, initial_time=mix_time)
type_A_md_temp_profile.add_state_point(cure_time, cure_kt)

out_dir = str('.')
sim_name = 'lj_NP'
out_dir = os.path.join(out_dir, sim_name)
myEpoxySim = es.ABCNPTypeEpoxyLJHarmonicSimulation(sim_name,
                                                   mix_time=mix_time,
                                                   mix_kt=mix_kt,
                                                   temp_prof=type_A_md_temp_profile,
                                                   bond=True,
                                                   n_mul=1.0,
                                                   shrink=True,
                                                   num_a=1000,
                                                   num_b=2000,
                                                   num_spheres=1,
                                                   AA_interaction=0.25,
                                                   AB_interation=0.05,
                                                   AC_interaction=0.05,
                                                   BC_interaction=0.05,
                                                   density=1.0,
                                                   shrink_time=1e5,
                                                   output_dir=out_dir,
                                                   use_dybond_plugin=True)

myEpoxySim.execute()

