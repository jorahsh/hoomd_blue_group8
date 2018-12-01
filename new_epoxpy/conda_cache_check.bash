#!/usr/bin/env bash

if [ -d /opt/conda/envs/epoxpy_env ]; then
	echo "Using Cache";
	source activate epoxpy_env 
else
	echo "Rebuilding Conda Env";
	conda env create -f environment.yml;
        source activate epoxpy_env
fi
