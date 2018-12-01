import pytest


class BaseTest:

    @pytest.fixture
    def a(self):
        from epoxpy.lib import A
        return A()

    @pytest.fixture
    def b(self):
        from epoxpy.lib import B
        return B()

    @pytest.fixture
    def c(self):
        from epoxpy.lib import C
        return C()

    @pytest.fixture
    def c10(self):
        from epoxpy.lib import C10
        return C10(rotate_random=False)

    @pytest.fixture
    def epoxy_a_10_b_20_c10_2_blend(self):
        from epoxpy.lib import Epoxy_A_10_B_20_C10_2_Blend
        return Epoxy_A_10_B_20_C10_2_Blend()

    @pytest.fixture
    def c10_new_position(self):
        from epoxpy.lib import C10
        return C10(c1_pos=[10, 0, 0], rotate_random=False)

    @pytest.fixture
    def c10_ref_xyz(self):
        array = ([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                  [-1.12757026e-17, 1.40000000e-01, 3.38491720e-18],
                  [-3.55618313e-17, 2.80000000e-01, 1.73362268e-17],
                  [-6.41847686e-17, 4.20000000e-01, 2.20807992e-17],
                  [-7.37257477e-17, 5.60000000e-01, 1.84827171e-17],
                  [-8.06646416e-17, 7.00000000e-01, 3.79288412e-17],
                  [-1.49186219e-16, 8.40000000e-01, 2.93897951e-17],
                  [-2.74953671e-16, 9.80000000e-01, -4.58905298e-18],
                  [-4.29344060e-16, 1.12000000e+00, 2.73277095e-18],
                  [-5.88938620e-16, 1.26000000e+00, 5.04984341e-17]])
        return array

    @pytest.fixture
    def c10_new_ref_xyz(self):
        array = ([[1.00000000e+01, 0.00000000e+00, 0.00000000e+00],
                  [1.00000000e+01, 1.40000000e-01, 3.38491720e-18],
                  [1.00000000e+01, 2.80000000e-01, 1.73362268e-17],
                  [1.00000000e+01, 4.20000000e-01, 2.20807992e-17],
                  [1.00000000e+01, 5.60000000e-01, 1.84827171e-17],
                  [1.00000000e+01, 7.00000000e-01, 3.79288412e-17],
                  [1.00000000e+01, 8.40000000e-01, 2.93897951e-17],
                  [1.00000000e+01, 9.80000000e-01, -4.58905298e-18],
                  [1.00000000e+01, 1.12000000e+00, 2.73277095e-18],
                  [1.00000000e+01, 1.26000000e+00, 5.04984341e-17]])
        return array
