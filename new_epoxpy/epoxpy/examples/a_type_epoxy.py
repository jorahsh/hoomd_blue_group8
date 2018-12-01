import epoxpy.a_type_epoxy_lj_harmonic_simulation as es
import epoxpy.temperature_profile_builder as tpb
import random
import os
import gsd.hoomd
import numpy as np
import epoxpy.common as cmn

random.seed(1020)

#mix_time = 1e3
#mix_kt = 1.5
#cure_kt = 1.5
#cure_time = 10000
#
#type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=cure_kt)
#type_A_md_temp_profile.add_state_point(cure_time, cure_kt)
#
#out_dir = str('.')
#sim_name = 'a_type'
#out_dir = os.path.join(out_dir, sim_name)
#myEpoxySim = es.ATypeEpoxyLJHarmonicSimulation(sim_name,
#                                               mix_time=mix_time,
#                                               mix_kt=mix_kt,
#                                               temp_prof=type_A_md_temp_profile,
#                                               bond=True, n_mul=300.0, shrink=True,
#                                               shrink_time=1e4,
#                                               mix_dt=1e-4,
#                                               md_dt=1e-2,
#                                               AA_interaction=0.25,
#                                               AA_bond_const=100.,
#                                               bond_radius=1.0,
#                                               dcd_write=100,
#                                               exclude_mixing_from_output=True,
#                                               integrator=cmn.Integrators.LANGEVIN.name,
#                                               output_dir=out_dir,
#                                               use_dybond_plugin=True)

mix_time = 1e3
mix_kt = 2.0
cure_kt = 1.3
cure_time = 10000

type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=cure_kt)
type_A_md_temp_profile.add_state_point(cure_time, cure_kt)

out_dir = str('')
sim_name = 'a_type'
out_dir = os.path.join(out_dir, sim_name)
myEpoxySim = es.ATypeEpoxyLJHarmonicSimulation(sim_name,
                                               mix_time=mix_time,
                                               mix_kt=mix_kt,
                                               temp_prof=type_A_md_temp_profile,
                                               bond=True, n_mul=600.0, shrink=True,
                                               shrink_time=1e4,
                                               mix_dt=1e-4,
                                               md_dt=1e-2,
                                               output_dir=out_dir,
                                               use_dybond_plugin=True)

myEpoxySim.execute()
