#!/bin/bash

# Create a new virtual environment named codeflow with Python 3.10
conda create -n codeflow python=3.10 -y

# Activate the virtual environment
source activate codeflow

# Install the required Python libraries
pip install -r requirements.txt