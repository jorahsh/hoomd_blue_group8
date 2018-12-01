from base_test import BaseTest
import pytest


class TestDyBondBonding(BaseTest):
    """
    Test class for testing simulation result for ABCTypeEpoxySimulation with
    dybond plugin. Requires hoomd with dybond plugin
    """

    @pytest.mark.long
    @pytest.mark.dybond_bonding
    def test_epoxy_sim_dpd_with_enthalpy(self, tmpdir):
        """
        Here we are testing if set point temperature increases when enthalpy change is enabled for dpd
        :param datadir:
        :param tmpdir:
        :return:
        """
        from epoxpy.abc_type_epoxy_simulation import ABCTypeEpoxySimulation
        import epoxpy.abc_type_epoxy_dpd_simulation as es
        import epoxpy.temperature_profile_builder as tpb
        import random
        import os
        import gsd.hoomd
        import numpy as np
        # from cme_utils.analyze import bond_dist

        random.seed(1020)

        mix_time = 3e2
        mix_kt = 2.0
        cure_kt = 2.0
        time_scale = 100
        deltaT = 1e-3
        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt,
                                                                     initial_time=mix_time)
        type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)

        out_dir = str(tmpdir)
        sim_name = 'shrunk_freud_bonding'
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.ABCTypeEpoxyDPDSimulation(sim_name,
                                                  mix_time=mix_time,
                                                  mix_kt=mix_kt,
                                                  temp_prof=type_A_md_temp_profile,
                                                  bond=True,
                                                  n_mul=2.0,
                                                  shrink=True,
                                                  shrink_time=1e2,
                                                  enable_rxn_enthalpy=True,
                                                  percent_bonds_per_step=100,
                                                  bond_period=1,
                                                  deltaT=deltaT,
                                                  output_dir=out_dir,
                                                  use_dybond_plugin=True)

        myEpoxySim.execute()

        current_gsd = tmpdir.join(sim_name, 'data.gsd')
        gsd_path = str(current_gsd)
        print('reading gsd: ', gsd_path)
        f = gsd.fl.GSDFile(gsd_path, 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snapshot = t[-1]
        current_bonds = snapshot.bonds.N
        assert snapshot.particles.N == 100
        print('test_epoxy_sim_dpd_with_enthalpy. current:{}'.format(current_bonds))
        assert current_bonds > 1  # Just checking if some bonds are being made

        # this test works fine if cme_utils can be installed from pipelines
        # b_distances = bond_dist.get_bond_distances(snapshot,'A-B')
        # m, s = bond_dist.get_bond_distribution(b_distances)
        # assert np.isclose(m,1.0,rtol=0.2)

        ab_bonds = snapshot.bonds.typeid[snapshot.bonds.typeid == 1]
        num_ab_bonds = len(ab_bonds)
        predicted_kT = cure_kt + (deltaT * num_ab_bonds)
        assert (np.isclose(predicted_kT, myEpoxySim.cure_kt, rtol=deltaT))

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

    @pytest.mark.long
    @pytest.mark.dybond_bonding
    def test_epoxy_sim_lj_harmonic_with_enthalpy(self, tmpdir):
        """
        Here we are testing if set point temperature increases when enthalpy change is enabled for lj harmonic
        :param datadir:
        :param tmpdir:
        :return:
        """
        from epoxpy.abc_type_epoxy_simulation import ABCTypeEpoxySimulation
        import epoxpy.abc_type_epoxy_lj_harmonic_simulation as es
        import epoxpy.temperature_profile_builder as tpb
        import random
        import os
        import gsd.hoomd
        import numpy as np
        import epoxpy.common as cmn

        random.seed(1020)

        mix_time = 1e4
        mix_kt = 2.0
        cure_kt = 2.0
        time_scale = 100
        n_mul = 5.0
        n_part = n_mul * 50
        deltaT = 1e-3
        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt,
                                                                     initial_time=mix_time)
        type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)

        out_dir = str(tmpdir)
        sim_name = 'shrunk_freud_bonding'
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.ABCTypeEpoxyLJHarmonicSimulation(sim_name,
                                                         mix_time=mix_time,
                                                         mix_kt=mix_kt,
                                                         temp_prof=type_A_md_temp_profile,
                                                         bond=True,
                                                         n_mul=n_mul,
                                                         shrink=True,
                                                         shrink_time=1e4,
                                                         mix_dt=1e-4,
                                                         md_dt=1e-2,
                                                         integrator=cmn.Integrators.LANGEVIN.name,
                                                         output_dir=out_dir,
                                                         use_dybond_plugin=True,
                                                         enable_rxn_enthalpy=True,
                                                         percent_bonds_per_step=100,
                                                         bond_period=1,
                                                         deltaT=deltaT,
                                                         density=0.01)

        myEpoxySim.execute()

        current_gsd = tmpdir.join(sim_name, 'data.gsd')
        gsd_path = str(current_gsd)
        print('reading gsd: ', gsd_path)
        f = gsd.fl.GSDFile(gsd_path, 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snapshot = t[-1]
        current_bonds = snapshot.bonds.N
        assert snapshot.particles.N == n_part
        print('test_epoxy_sim_freud_shrunk_regression. current:{}'.format(current_bonds))
        assert current_bonds >= 1  # Just checking if some bonds are being made

        ab_bonds = snapshot.bonds.typeid[snapshot.bonds.typeid == 1]
        num_ab_bonds = len(ab_bonds)
        predicted_kT = cure_kt + (deltaT * num_ab_bonds)
        assert (np.isclose(predicted_kT, myEpoxySim.cure_kt, rtol=deltaT))

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

    @pytest.mark.long
    @pytest.mark.dybond_bonding
    def test_epoxy_sim_dybond_regression(self,  tmpdir):
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
        import gsd.hoomd
        import numpy as np
        #from cme_utils.analyze import bond_dist

        random.seed(1020)

        mix_time = 3e2
        mix_kt = 2.0
        cure_kt = 2.0
        time_scale = 100
        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt, initial_time=mix_time)
        type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)

        out_dir = str(tmpdir)
        sim_name = 'shrunk_freud_bonding'
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.ABCTypeEpoxyDPDSimulation(sim_name, mix_time=mix_time, mix_kt=mix_kt,
                                               temp_prof=type_A_md_temp_profile,
                                               bond=True, n_mul=2.0, shrink=True,
                                               shrink_time=1e2,
                                               output_dir=out_dir,
                                               use_dybond_plugin=True)

        myEpoxySim.execute()

        current_gsd = tmpdir.join(sim_name, 'data.gsd')
        gsd_path = str(current_gsd)
        print('reading gsd: ', gsd_path)
        f = gsd.fl.GSDFile(gsd_path, 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snapshot = t[-1]
        current_bonds = snapshot.bonds.N
        assert snapshot.particles.N == 100
        print('test_epoxy_sim_freud_shrunk_regression. current:{}'.format(current_bonds))
        assert current_bonds > 1 #Just checking if some bonds are being made

        #this test works fine if cme_utils can be installed from pipelines
        #b_distances = bond_dist.get_bond_distances(snapshot,'A-B')
        #m, s = bond_dist.get_bond_distribution(b_distances)
        #assert np.isclose(m,1.0,rtol=0.2)

        idxs, counts = np.unique(snapshot.bonds.group, return_counts=True)
        print('########################idxs',idxs,counts)
        print(snapshot.bonds.group)
        for index,idx  in enumerate(idxs):
            p_typeid = snapshot.particles.typeid[idx]
            p_type = snapshot.particles.types[p_typeid]
            if p_type == 'A':
                assert (counts[index] <= myEpoxySim.max_a_bonds)
            elif p_type == 'B':
                assert (counts[index] <= myEpoxySim.max_b_bonds)

    @pytest.mark.long
    @pytest.mark.dybond_bonding
    def test_a_type_epoxy_sim_dybond_lj_harmonic_langevin(self,  tmpdir):
        """
        Here we are doing regression testing for the new bonding routine that operates and the mbuild initial structure
        whose volume is shrunk to a density of 1.0
        :param datadir:
        :param tmpdir:
        :return:
        """
        import epoxpy.a_type_epoxy_lj_harmonic_simulation as es
        import epoxpy.temperature_profile_builder as tpb
        import random
        import os
        import gsd.hoomd
        import numpy as np

        random.seed(1020)

        mix_time = 1e3
        mix_kt = 2.0
        cure_kt = 1.3
        cure_time = 1e4
        n_mol = 5

        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=cure_kt)
        type_A_md_temp_profile.add_state_point(cure_time, cure_kt)

        out_dir = str(tmpdir)
        sim_name = 'shrunk_freud_bonding'
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.ATypeEpoxyLJHarmonicSimulation(sim_name,
                                                       mix_time=mix_time,
                                                       mix_kt=mix_kt,
                                                       temp_prof=type_A_md_temp_profile,
                                                       bond=True, n_mul=n_mol, shrink=True,
                                                       shrink_time=1e4,
                                                       mix_dt=1e-4,
                                                       md_dt=1e-2,
                                                       output_dir=out_dir,
                                                       use_dybond_plugin=True,
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

    @pytest.mark.long
    @pytest.mark.dybond_bonding
    def test_epoxy_sim_dybond_lj_harmonic_langevin(self,  tmpdir):
        """
        Here we are doing regression testing for the new bonding routine that operates and the mbuild initial structure
        whose volume is shrunk to a density of 1.0
        :param datadir:
        :param tmpdir:
        :return:
        """
        from epoxpy.abc_type_epoxy_simulation import ABCTypeEpoxySimulation
        import epoxpy.abc_type_epoxy_lj_harmonic_simulation as es
        import epoxpy.temperature_profile_builder as tpb
        import random
        import os
        import gsd.hoomd
        import numpy as np
        import epoxpy.common as cmn

        random.seed(1020)

        mix_time = 1e4
        mix_kt = 2.0
        cure_kt = 2.0
        time_scale = 100
        n_mul = 5.0
        n_part = n_mul*50
        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt, initial_time=mix_time)
        type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)

        out_dir = str(tmpdir)
        sim_name = 'shrunk_freud_bonding'
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.ABCTypeEpoxyLJHarmonicSimulation(sim_name, mix_time=mix_time, mix_kt=mix_kt,
                                               temp_prof=type_A_md_temp_profile,
                                               bond=True, n_mul=n_mul, shrink=True,
                                               shrink_time=1e4,
                                               mix_dt=1e-4,
                                               md_dt=1e-2,
                                               integrator=cmn.Integrators.LANGEVIN.name,
                                               output_dir=out_dir,
                                               use_dybond_plugin=True,
                                               density=0.01)

        myEpoxySim.execute()

        current_gsd = tmpdir.join(sim_name, 'data.gsd')
        gsd_path = str(current_gsd)
        print('reading gsd: ', gsd_path)
        f = gsd.fl.GSDFile(gsd_path, 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snapshot = t[-1]
        current_bonds = snapshot.bonds.N
        assert snapshot.particles.N == n_part
        print('test_epoxy_sim_freud_shrunk_regression. current:{}'.format(current_bonds))
        assert current_bonds >= 1 #Just checking if some bonds are being made

        idxs, counts = np.unique(snapshot.bonds.group, return_counts=True)
        print('########################idxs',idxs,counts)
        print(snapshot.bonds.group)
        for index,idx  in enumerate(idxs):
            p_typeid = snapshot.particles.typeid[idx]
            p_type = snapshot.particles.types[p_typeid]
            if p_type == 'A':
                assert (counts[index] <= myEpoxySim.max_a_bonds)
            elif p_type == 'B':
                assert (counts[index] <= myEpoxySim.max_b_bonds)

    @pytest.mark.long
    @pytest.mark.dybond_bonding
    def test_epoxy_sim_dybond_lj_harmonic_npt(self,  tmpdir):
        """
        Here we are doing regression testing for the new bonding routine that operates and the mbuild initial structure
        whose volume is shrunk to a density of 1.0
        :param datadir:
        :param tmpdir:
        :return:
        """
        from epoxpy.abc_type_epoxy_simulation import ABCTypeEpoxySimulation
        import epoxpy.abc_type_epoxy_lj_harmonic_simulation as es
        import epoxpy.temperature_profile_builder as tpb
        import random
        import os
        import gsd.hoomd
        import numpy as np
        import epoxpy.common as cmn

        random.seed(1020)

        mix_time = 1e3
        mix_kt = 2.0
        cure_kt = 2.0
        time_scale = 100
        n_mul = 6.0
        n_part = n_mul*50
        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt, initial_time=mix_time)
        type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)

        out_dir = str(tmpdir)
        sim_name = 'shrunk_freud_bonding'
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.ABCTypeEpoxyLJHarmonicSimulation(sim_name, mix_time=mix_time, mix_kt=mix_kt,
                                                         temp_prof=type_A_md_temp_profile,
                                                         bond=True, n_mul=n_mul, shrink=True,
                                                         shrink_time=1e4,
                                                         mix_dt=1e-4,
                                                         md_dt=1e-2,
                                                         integrator=cmn.Integrators.NPT.name,
                                                         output_dir=out_dir,
                                                         use_dybond_plugin=True,
                                                         density=0.01)

        myEpoxySim.execute()

        current_gsd = tmpdir.join(sim_name, 'data.gsd')
        gsd_path = str(current_gsd)
        print('reading gsd: ', gsd_path)
        f = gsd.fl.GSDFile(gsd_path, 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snapshot = t[-1]
        current_bonds = snapshot.bonds.N
        assert snapshot.particles.N == n_part
        print('test_epoxy_sim_freud_shrunk_regression. current:{}'.format(current_bonds))
        assert current_bonds >= 1  # Just checking if some bonds are being made

        idxs, counts = np.unique(snapshot.bonds.group, return_counts=True)
        print('########################idxs',idxs,counts)
        print(snapshot.bonds.group)
        for index,idx  in enumerate(idxs):
            p_typeid = snapshot.particles.typeid[idx]
            p_type = snapshot.particles.types[p_typeid]
            if p_type == 'A':
                assert (counts[index] <= myEpoxySim.max_a_bonds)
            elif p_type == 'B':
                assert (counts[index] <= myEpoxySim.max_b_bonds)

    @pytest.mark.long
    @pytest.mark.dybond_bonding
    def test_epoxy_sim_dybond_dpdlj(self,  tmpdir):
        """
        Here we are doing regression testing for the new bonding routine that operates and the mbuild initial structure
        whose volume is shrunk to a density of 3.0
        :param datadir:
        :param tmpdir:
        :return:
        """
        from epoxpy.abc_type_epoxy_simulation import ABCTypeEpoxySimulation
        import epoxpy.abc_type_epoxy_dpdlj_simulation as es
        import epoxpy.temperature_profile_builder as tpb
        import random
        import os
        import gsd.hoomd
        import numpy as np

        random.seed(1020)

        mix_time = 1e3
        mix_kt = 2.0
        cure_kt = 2.0
        time_scale = 100
        n_mul = 6.0
        n_part = n_mul*50
        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt, initial_time=mix_time)
        type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)

        out_dir = str(tmpdir)
        sim_name = 'shrunk_freud_bonding'
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.ABCTypeEpoxyDPDLJSimulation(sim_name, mix_time=mix_time, mix_kt=mix_kt,
                                               temp_prof=type_A_md_temp_profile,
                                               bond=True, n_mul=n_mul, shrink=True,
                                               shrink_time=5e4,
                                               mix_dt=1e-4,
                                               md_dt=1e-2,
                                               output_dir=out_dir,
                                               use_dybond_plugin=True)

        myEpoxySim.execute()

        current_gsd = tmpdir.join(sim_name, 'data.gsd')
        gsd_path = str(current_gsd)
        print('reading gsd: ', gsd_path)
        f = gsd.fl.GSDFile(gsd_path, 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snapshot = t[-1]
        current_bonds = snapshot.bonds.N
        assert snapshot.particles.N == n_part
        print('test_epoxy_sim_freud_shrunk_regression. current:{}'.format(current_bonds))
        assert current_bonds >= 1 #Just checking if some bonds are being made

        idxs, counts = np.unique(snapshot.bonds.group, return_counts=True)
        print('########################idxs',idxs,counts)
        print(snapshot.bonds.group)
        for index,idx  in enumerate(idxs):
            p_typeid = snapshot.particles.typeid[idx]
            p_type = snapshot.particles.types[p_typeid]
            if p_type == 'A':
                assert (counts[index] <= myEpoxySim.max_a_bonds)
            elif p_type == 'B':
                assert (counts[index] <= myEpoxySim.max_b_bonds)

    @pytest.mark.long
    @pytest.mark.dybond_bonding
    def test_epoxy_sim_dybond_dpdfene(self,  tmpdir):
        """
        Here we are doing regression testing for the new bonding routine that operates and the mbuild initial structure
        whose volume is shrunk to a density of 3.0
        :param datadir:
        :param tmpdir:
        :return:
        """
        from epoxpy.abc_type_epoxy_simulation import ABCTypeEpoxySimulation
        import epoxpy.abc_type_epoxy_dpdfene_simulation as es
        import epoxpy.temperature_profile_builder as tpb
        import random
        import os
        import gsd.hoomd
        import numpy as np

        random.seed(1020)

        mix_time = 1e3
        mix_kt = 2.0
        cure_kt = 2.0
        time_scale = 1
        n_mul = 6
        n_part = n_mul * 50
        type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt, initial_time=mix_time)
        type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)

        out_dir = str(tmpdir)
        sim_name = 'shrunk_freud_bonding'
        out_dir = os.path.join(out_dir, sim_name)
        myEpoxySim = es.ABCTypeEpoxyDPDFENESimulation(sim_name, mix_time=mix_time, mix_kt=mix_kt,
                                               temp_prof=type_A_md_temp_profile,
                                               bond=True, n_mul=n_mul, shrink=True,
                                               shrink_time=5e4,
                                               mix_dt=1e-4,
                                               md_dt=1e-2,
                                               output_dir=out_dir,
                                               use_dybond_plugin=True)

        myEpoxySim.execute()

        current_gsd = tmpdir.join(sim_name, 'data.gsd')
        gsd_path = str(current_gsd)
        print('reading gsd: ', gsd_path)
        f = gsd.fl.GSDFile(gsd_path, 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snapshot = t[-1]
        current_bonds = snapshot.bonds.N
        assert snapshot.particles.N == n_part
        print('test_epoxy_sim_freud_shrunk_regression. current:{}'.format(current_bonds))
        assert current_bonds >= 1#Just checking if some bonds are being made

        idxs, counts = np.unique(snapshot.bonds.group, return_counts=True)
        print('########################idxs',idxs,counts)
        print(snapshot.bonds.group)
        for index,idx  in enumerate(idxs):
            p_typeid = snapshot.particles.typeid[idx]
            p_type = snapshot.particles.types[p_typeid]
            if p_type == 'A':
                assert (counts[index] <= myEpoxySim.max_a_bonds)
            elif p_type == 'B':
                assert (counts[index] <= myEpoxySim.max_b_bonds)


#----------================== This test needs a the mbuild.fill_box parameter fix_orientation ============------------
    #@pytest.mark.long
    #@pytest.mark.dybond_bonding
    #def test_epoxy_sim_NP_lj_harmonic(self,  tmpdir):
    #    """
    #    Here we are doing regression testing for the new bonding routine that operates and the mbuild initial structure
    #    whose volume is shrunk to a density of 3.0
    #    :param datadir:
    #    :param tmpdir:
    #    :return:
    #    """
    #    import epoxpy.abc_NP_type_epoxy_lj_harmonic_simulation as es
    #    import epoxpy.temperature_profile_builder as tpb
    #    import epoxpy.bonding as bondClass
    #    import random
    #    import os
    #    import gsd.hoomd
    #    import numpy as np
#
    #    random.seed(1020)
#
    #    mix_time = 1e3
    #    mix_kt = 2.0
    #    cure_kt = 2.0
    #    time_scale = 100
    #    temp_scale = 1
    #    type_A_md_temp_profile = tpb.LinearTemperatureProfileBuilder(initial_temperature=mix_kt, initial_time=mix_time)
    #    type_A_md_temp_profile.add_state_point(500 * time_scale, cure_kt)
#
    #    out_dir = str(tmpdir)
    #    sim_name = 'NP_lj_harmonic'
    #    out_dir = os.path.join(out_dir, sim_name)
    #    myEpoxySim = es.ABCNPTypeEpoxyLJHarmonicSimulation(sim_name, mix_time=mix_time, mix_kt=mix_kt,
    #                                   temp_prof=type_A_md_temp_profile,
    #                                   bond=True, n_mul=2.0, shrink=True,
    #                                   shrink_time=1e2,
    #                                   output_dir=out_dir,
    #                                   use_dybond_plugin=True)
#
    #    myEpoxySim.execute()
#
    #    current_gsd = tmpdir.join(sim_name, 'data.gsd')
    #    gsd_path = str(current_gsd)
    #    print('reading gsd: ', gsd_path)
    #    f = gsd.fl.GSDFile(gsd_path, 'rb')
    #    t = gsd.hoomd.HOOMDTrajectory(f)
    #    snapshot = t[-1]
    #    current_bonds = snapshot.bonds.N
    #    assert snapshot.particles.N == 720
    #    print('test_epoxy_sim_freud_shrunk_regression. current:{}'.format(current_bonds))
    #    assert current_bonds > 30 #Just checking if some bonds are being made
#
    #    idxs, counts = np.unique(snapshot.bonds.group, return_counts=True)
    #    print('########################idxs',idxs,counts)
    #    print(snapshot.bonds.group)
    #    for index,idx  in enumerate(idxs):
    #        p_typeid = snapshot.particles.typeid[idx]
    #        p_type = snapshot.particles.types[p_typeid]
    #        if p_type == 'A':
    #            assert (counts[index] <= bondClass.FreudBonding.max_a_bonds)
    #        elif p_type == 'B':
    #            assert (counts[index] <= bondClass.FreudBonding.max_b_bonds)
