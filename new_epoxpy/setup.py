from __future__ import print_function


from setuptools import setup, find_packages

setup(name='epoxpy',
        version='2.0.2',
        description='Simulation wrappers for epoxy simulations using HOOMD',
        url='https://bitbucket.org/cmelab/epoxpy',
        author='cmelab',
        author_email='stephenthomas1@boisestate.edu',
        license='GPL-3.0',
        packages=find_packages(),
        package_dir={'epoxpy':'epoxpy'},
#        install_requires=[
#            'numpy',
#            'matplotlib',
#            'mbuild',
#            'pytest'
#        ],
        zip_safe=False)
