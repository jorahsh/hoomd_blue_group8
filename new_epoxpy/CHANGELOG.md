10, Jan 2018; v 2.0.3
---------------------
Author: Stephen Thomas
Changes:
1) Restructures epoxy_simulation inheritance structure to make extensions easier and added abstract methods to enable hassle free extensions. See for example "get_log_quantities"
2) Added simulations classes and tests for lj-harmonic, lj-fene and dpdlj-harmonic and dpd-lj-fene combinations 
3) Added NPT integrator option for lj-harmonic
4) Works well with hoomd_dybond v.2.2.1.1
