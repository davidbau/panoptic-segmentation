#!/usr/bin/env bash

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

if [ ! -e ~/.conda/envs/lenv ]
then
    setup/setup_lenv.sh
fi
source activate lenv

notebooks/setup_notebooks.sh
UPSNet/init.sh
