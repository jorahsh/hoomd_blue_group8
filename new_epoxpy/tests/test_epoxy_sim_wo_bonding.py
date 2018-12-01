from base_test import BaseTest
import pytest


class TestWOBonding(BaseTest):
    """
    Test class for testing simulation result for ABCTypeEpoxySimulation with baseline simulation result.
    Checks if positions of particles are close to baseline particle positions.
    """
    @pytest.mark.long
    def test_epoxy_sim_wo_bonding(self,  tmpdir):
        import epoxpy.abc_type_epoxy_dpd_simulation as es
        import epoxpy.temperature_profile_builder as tpb
        import random
        import os
        import gsd.hoomd

        random.seed(1020)
        print('\n# Test: test_epoxy_sim_wo_bonding')

        mix_time = 3e4
        mix_kt = 2.0
        cure_kt = 2.0
        time_scale = 100
        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt, initial_time=mix_time)
        type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)

        sim_name = 'wo_bonding'
        out_dir = str(tmpdir)
        exclude_mixing_in_output = True
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.ABCTypeEpoxyDPDSimulation(sim_name,
                                                  mix_time=mix_time,
                                                  mix_kt=mix_kt,
                                                  temp_prof=type_A_md_temp_profile,
                                                  output_dir=out_dir,
                                                  n_mul=2.0,
                                                  exclude_mixing_in_output=exclude_mixing_in_output,
                                                  shrink=False)

        myEpoxySim.execute()

        current_gsd = tmpdir.join(sim_name, 'data.gsd')
        gsd_path = str(current_gsd)
        f = gsd.fl.GSDFile(gsd_path, 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snapshot = t[-1]

        assert snapshot.particles.N == 100

    @pytest.mark.long
    @pytest.mark.dybond_bonding
    def test_neat_toughener_lj_harmonic_langevin(self,  tmpdir):
        """
        Testing if the neat toughener simulation works fine and continues so..
        :param datadir:
        :param tmpdir:
        :return:
        """
        import epoxpy.neat_toughener_lj_harmonic_simulation as es
        import epoxpy.temperature_profile_builder as tpb
        import random
        import os
        import gsd.hoomd
        import numpy as np

        random.seed(1020)

        mix_time = 1e4
        mix_kt = 2.0
        cure_kt = 1.3
        cure_time = 1e4
        cXX=100
        num_cXX=10
        n_mol = cXX*num_cXX
        num_bonds_expected = (cXX-1)*num_cXX

        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=cure_kt)
        type_A_md_temp_profile.add_state_point(cure_time, cure_kt)

        out_dir = str(tmpdir)
        sim_name = 'shrunk_freud_bonding'
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.NeatToughenerLJHarmonicSimulation(sim_name,
                                                          mix_time=mix_time,
                                                          mix_kt=mix_kt,
                                                          temp_prof=type_A_md_temp_profile,
                                                          shrink_time=1e4,
                                                          mix_dt=1e-4,
                                                          md_dt=1e-2,
                                                          num_cXX=num_cXX,
                                                          cXX=cXX,
                                                          output_dir=out_dir,
                                                          density=0.01)

        myEpoxySim.execute()

        current_gsd = tmpdir.join(sim_name, 'data.gsd')
        gsd_path = str(current_gsd)
        print('reading gsd: ', gsd_path)
        f = gsd.fl.GSDFile(gsd_path, 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snapshot = t[-1]
        current_bonds = snapshot.bonds.N
        assert snapshot.particles.N == n_mol
        print('test_epoxy_sim_freud_shrunk_regression. current:{}'.format(current_bonds))
        assert current_bonds == num_bonds_expected  # Just checking if some bonds are there

        idxs, counts = np.unique(snapshot.bonds.group, return_counts=True)
        print('########################idxs', idxs, counts)
        print(snapshot.bonds.group)
        for index, idx in enumerate(idxs):
            p_typeid = snapshot.particles.typeid[idx]
            p_type = snapshot.particles.types[p_typeid]
            assert  (p_type == 'C')
            assert (counts[index] <= 2)