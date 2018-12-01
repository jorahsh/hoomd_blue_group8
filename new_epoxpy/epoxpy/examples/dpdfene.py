import epoxpy.abc_type_epoxy_dpdfene_simulation as es
import epoxpy.temperature_profile_builder as tpb
import epoxpy.bonding as bondClass
import random
import os
import gsd.hoomd
import numpy as np

random.seed(1020)

mix_time = 1e3
mix_kt = 2.0
cure_kt = 2.0
time_scale = 100
temp_scale = 1
type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt, initial_time=mix_time)
type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)

out_dir = str('.')
sim_name = 'shrunk_freud_bonding'
out_dir = os.path.join(out_dir, sim_name)
myEpoxySim = es.ABCTypeEpoxyDPDFENESimulation(sim_name, mix_time=mix_time, mix_kt=mix_kt,
                                       temp_prof=type_A_md_temp_profile,
                                       bond=True, n_mul=4.0, shrink=True,
                                       shrink_time=1e4,
                                       mix_dt=1e-4,
                                       md_dt=1e-2,
                                       output_dir=out_dir,
                                       use_dybond_plugin=True)

myEpoxySim.execute()

