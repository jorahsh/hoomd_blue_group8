from base_test import BaseTest
import pytest

class TestSimRestart(BaseTest):
    """
    Test class for testing simulation result for ABCTypeEpoxySimulation with
    dybond plugin. Requires hoomd with dybond plugin
    """
    @pytest.mark.long
    @pytest.mark.dybond_bonding
    def test_restart(self,  tmpdir):
        """
        Here we are doing regression testing for the new bonding routine that operates and the mbuild initial structure
        whose volume is shrunk to a density of 3.0
        :param datadir:
        :param tmpdir:
        :return:
        """
        from epoxpy.abc_type_epoxy_simulation import ABCTypeEpoxySimulation
        import epoxpy.abc_type_epoxy_dpd_simulation as es
        import epoxpy.temperature_profile_builder as tpb
        import random
        import os
        import numpy as np
        import gsd.hoomd
        #from cme_utils.analyze import bond_dist

        random.seed(1020)

        mix_time = 3e2
        mix_kt = 2.0
        cure_kt = 4.0
        t1 = 25000
        t2 = 25000
        total_time = t1+t2

        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=cure_kt, initial_time=0)
        type_A_md_temp_profile.add_state_point(t1, cure_kt)
        n_mul = 1.0
        n_part = n_mul * 13

        out_dir = str(tmpdir)
        sim_name = 'shrunk_freud_bonding'
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.ABCTypeEpoxyDPDSimulation(sim_name,
                                                  mix_time=mix_time,
                                                  mix_kt=mix_kt,
                                                  temp_prof=type_A_md_temp_profile,
                                                  bond=True,
                                                  n_mul=n_mul,
                                                  bond_period=1,
                                                  percent_bonds_per_step=100.0,
                                                  activation_energy=0.005,
                                                  sec_bond_weight=0.01,
                                                  shrink=True,
                                                  shrink_time=1e2,
                                                  output_dir=out_dir,
                                                  use_dybond_plugin=True,
                                                  num_a=1,
                                                  num_b=2,
                                                  num_c10=1)

        myEpoxySim.execute()
        # run for longer at cure_kt
        type_A_md_temp_profile.add_state_point(t2, cure_kt)
        # now update the temp prof in the simulation class
        myEpoxySim.temp_prof = type_A_md_temp_profile
        # run the simulation
        myEpoxySim.execute()

        current_gsd = os.path.join(out_dir, 'data.gsd')
        gsd_path = str(current_gsd)
        print('reading gsd: ', gsd_path)
        f = gsd.fl.GSDFile(gsd_path, 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snapshot = t[-1]
        current_bonds = snapshot.bonds.N
        print(n_part, snapshot.particles.N)
        assert snapshot.particles.N == n_part
        print('test_epoxy_sim_freud_shrunk_regression. current:{}'.format(current_bonds))
        assert current_bonds >= 1  # Just checking if some bonds are being made

        idxs, counts = np.unique(snapshot.bonds.group, return_counts=True)
        print('########################idxs', idxs, counts)
        print(snapshot.bonds.group)
        for index, idx in enumerate(idxs):
            p_typeid = snapshot.particles.typeid[idx]
            p_type = snapshot.particles.types[p_typeid]
            if p_type == 'A':
                assert (counts[index] <= myEpoxySim.max_a_bonds)
            elif p_type == 'B':
                assert (counts[index] <= myEpoxySim.max_b_bonds)

        # First lets assert all the output files exist
        # And lets check if the first & last step is right for the msd & log
        output_files = ["data.gsd", "epoxpy_metadata.json", "initial.hoomdxml", "mixed.hoomdxml",
                        "msd.log", "out.log", "restart.gsd", "start.hoomdxml",
                        "start.hoomdxml", "traj.dcd"]

        for output_file in output_files:
            f_path = str(tmpdir.join(sim_name, output_file))
            assert os.path.isfile(f_path)

            if output_file == "msd.log":
                data = np.genfromtxt(f_path, names=True)
                time_steps = data['timestep']
                assert time_steps[0] == 0
                assert time_steps[-1] == total_time

            if output_file == "out.log":
                data = np.genfromtxt(f_path, names=True)
                time_steps = data['timestep']
                assert time_steps[0] == 0
                assert time_steps[-1] == total_time
                cure_percents = data['bond_percentAB']
                strictly_increasing = all(x <= y for x, y in zip(cure_percents, cure_percents[1:]))
                assert strictly_increasing
