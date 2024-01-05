#!/bin/bash

# Switch on echoing
set -v

# Installing packages
sudo apt -y install python3-pip
pip3 install jupyter
pip3 install findspark
pip3 install --upgrade nbconvert

# Add lines to .bashrc
if [[ $(grep "jupyter" ~/.bashrc) ]]; then
    echo "Your .bashrc already has the needed lines."
else
    cat .bashrc_jupyter >> ~/.bashrc
    source ~/.bashrc
fi

# Switch off echoing
set +v
