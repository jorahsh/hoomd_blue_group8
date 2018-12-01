import epoxpy.neat_toughener_lj_harmonic_simulation as es
import epoxpy.temperature_profile_builder as tpb
import random
import os
import gsd.hoomd
import numpy as np
import epoxpy.common as cmn

random.seed(1020)


mix_time = 1e4
mix_kt = 1.0
cure_kt = 1.3
cure_time = 10000

type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=cure_kt)
type_A_md_temp_profile.add_state_point(cure_time, cure_kt)

out_dir = str('')
sim_name = 'neat_tougheners'
out_dir = os.path.join(out_dir, sim_name)
myEpoxySim = es.NeatToughenerLJHarmonicSimulation(sim_name,
                                                          mix_time=mix_time,
                                                          mix_kt=mix_kt,
                                                          temp_prof=type_A_md_temp_profile,
                                                          shrink_time=1e4,
                                                          mix_dt=1e-4,
                                                          md_dt=1e-2,
                                                          num_cXX=10,
                                                          cXX=100,
                                                          output_dir=out_dir,
                                                          density=0.01)

myEpoxySim.execute()
