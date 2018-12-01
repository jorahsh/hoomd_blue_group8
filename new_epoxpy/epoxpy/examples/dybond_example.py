import epoxpy.abc_type_epoxy_simulation as es
import epoxpy.temperature_profile_builder as tpb
import os
import numpy as np

long_simulation = False

if long_simulation:
    time_scale = 10000
    mixing_time = 5e4
    n_mul = 1000.0
    t_Final = 7e6
    t_SetT = 0.75e6
    curing_log_period = 1e5
    log_write_period = 1e5
    data_write_period = 1e5
    stop_bonding_after = None # timesteps after start of curing
else:
    time_scale = 1
    mixing_time = 10
    n_mul = 100.0
    t_Final = 7e3
    t_SetT = 0.75e2
    curing_log_period = 1e4
    log_write_period = 1e1
    data_write_period = 1e1
    stop_bonding_after = None # timesteps after start of curing

mix_kt = 10.0
activation_energy = 2.0
sec_bond_weight = 2.0 #twice the primary bond energy for secondary bond
stop_bonding_after_percent = 100.0

av_calibrationT = 439.3636#K
av_aij_DDS_DEGBA = 30.729
av_aij_DDS_PES = 25.003
av_aij_DEGBA_PES = 30.532

temperature_profile =  'iso'#'lin_ramp''step'
bonding_period = 10
percent_bonds_per_step = 0.005
kT =500/av_calibrationT
AB_bond_const=4.0
AB_bond_dist=0.0
CC_bond_const=4.0
CC_bond_dist=0.0

if temperature_profile is 'iso':
    temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=kT,
                                                            initial_time=mixing_time)
    temp_profile.add_state_point(t_Final, kT)
elif temperature_profile is 'lin_ramp':
    temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=300/calibrationT,
                                                            initial_time=mixing_time)
    temp_profile.add_state_point(t_Final, kT)
elif temperature_profile is 'step':
    temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=300/calibrationT,
                                                            initial_time=mixing_time)
    temp_profile.add_state_point(t_SetT, kT)
    temp_profile.add_state_point(t_Final * time_scale, kT)
output_dir = os.path.join('.',temperature_profile)
myEpoxySim = es.ABCTypeEpoxySimulation(temperature_profile,
                                       output_dir=output_dir,
                                       mix_time=mixing_time,
                                       mix_kt=mix_kt,
                                       temp_prof=temp_profile,
                                       bond=True,
                                       n_mul=n_mul,
                                       dcd_write=data_write_period,
                                       bond_period=bonding_period,
                                       activation_energy=activation_energy,
                                       sec_bond_weight=sec_bond_weight,
                                       stop_after_percent=stop_bonding_after_percent,
                                       percent_bonds_per_step=percent_bonds_per_step,
                                       AA_interaction=25.00,
                                       AB_interaction=av_aij_DDS_DEGBA,
                                       AC_interaction=av_aij_DDS_PES,
                                       BC_interaction=av_aij_DEGBA_PES,
                                       AB_bond_const=AB_bond_const,
                                       AB_bond_dist= AB_bond_dist,
                                       CC_bond_const=CC_bond_const,
                                       CC_bond_dist= CC_bond_dist)


myEpoxySim.execute()

