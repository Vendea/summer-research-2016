#!/bin/bash

sudo apt-get install python-pip python-dev
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
#export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
sudo pip install mpi4py
git config --global user.email "tdhst231@mail.rmu.edu"
git config --global user.name "Trae Hurley"
cd
cd .ssh 
ssh-keygen -t dsa

