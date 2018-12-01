import mbuild as mb
from base_test import BaseTest
import numpy as np


class TestBaseCompounds(BaseTest):

    def test_a(self, a):
        assert a.n_bonds == 0
        assert a.n_particles == 1

    def test_b(self, b):
        assert b.n_bonds == 0
        assert b.n_particles == 1

    def test_c(self, c):
        assert c.n_bonds == 0
        assert c.n_particles == 1

    def test_c10(self, c10):
        assert c10.n_bonds == 9
        assert c10.n_particles == 10

#    def test_c10_negative_spacing(self):
#        with pytest.raises(ValueError) as excinfo:
#            from epoxpy.lib import C10
#            C10(spacing=-1.0)
#        assert 'Molecule type C10 expects a positive non zero spacing.' in str(excinfo.value)

    def test_c10_position(self, c10, c10_ref_xyz):
        assert np.allclose(c10.xyz, c10_ref_xyz)

    def test_c10_new_position(self, c10_new_position, c10_new_ref_xyz):
        assert np.allclose(c10_new_position.xyz, c10_new_ref_xyz)

    def test_batch_add(self, a, b, c10):
        compound = mb.Compound()
        compound.add([a, b, c10])
        assert compound.n_particles == 1 + 1 + 10
        assert compound.n_bonds == 0 + 0 + 9

    def test_epoxy_blend_10a_20b_2c10(self, epoxy_a_10_b_20_c10_2_blend):
        assert epoxy_a_10_b_20_c10_2_blend.n_bonds == 18
        assert epoxy_a_10_b_20_c10_2_blend.n_particles == 50
