#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate $1
conda install pip pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=12.1 jupyter=1.0.0 matplotlib=3.8.4 h5py=3.11.0 pytorch-cluster=1.6.3 -c pytorch -c nvidia -c pyg
python -m pip install torch_geometric==2.5.3 mmap-ninja==0.7.4 einops==0.8.0 scipy==1.13.0