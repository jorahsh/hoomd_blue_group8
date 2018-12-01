##HOW TO BUILD AND RUN ON R2

Go into the hoomd_jobs directory.

The first script you will want to run is:

$ sbatch step_01_python_setup.job

This will initialize your python environment.

Next, you will want to run the script:

$ sbatch compile.job

This will compile the hoomd_blue program.

Next, you will want to run the script:

$ sbatch validate_build.job

This just checks to make sure the program did in fact compile.

Finally, you will want to run the script:

$ sbatch epoxpy_test.job

This is the actual test of hoomd_blue using epoxpy.

When you want to want to clean your build, run:

$ sbatch clean.job 
