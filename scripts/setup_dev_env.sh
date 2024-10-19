#!/usr/bin/env bash
# ---------------------------------------------------
# Shell script to setup development environment
# ---------------------------------------------------
# steps:
# - setup virtualenv
# - install dependencies
# ---------------------------------------------------

# change to project root
cd ../
# setup virtualenv with dependencies with pip
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
# printf ">>\nVirtualenv created.\n"
# printf "\n >> \n\trun pre-commit install"

# OR:

# setup env and dependencies with conda
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
conda env create -f environment.yml
rm Miniforge3-$(uname)-$(uname -m).sh
printf ">>\nconda environment created.\n"
printf "\n >> \n\trun conda activate hems-rl && pre-commit install"
