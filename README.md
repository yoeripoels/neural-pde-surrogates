# Neural PDE Surrogates
This repository contains code for training neural PDE surrogates, fast neural network-based surrogate models that approximate PDE solutions. All models are implemented using PyTorch. 

## Setup
The code was tested using Python 3.9 and PyTorch 2.3.0. For ease of installation (and package versions) we provide `install_env.sh`, to be used with a conda environment:

```
conda create --name env_name python=3.9.5
source install_env.sh env_name
``` 


## Two-phase flows
This repository contains code to train models from the paper [Accelerating Simulation of Two-Phase Flows with Neural PDE Surrogates](https://openreview.net/forum?id=yIqszw9RUc&noteId=tSpWiDVORr) *(ICML 2024 AI4Science workshop)*

The datasets of the oil-expulsion problem, both for the case with and without obstacles, can be found [here](https://drive.google.com/drive/folders/13uwoIdzIXWJFeo1s5Zwaue5X290fMs7O). These should be extracted in `/data/`, e.g. such that we get `/data/twophase/snapshots.npy`

The configs are found in `/configs/train/cfg_twophase_*.py` for model  `*`  ∈ `{'drn', 'ufno', 'ufno_fno', 'unet'}`

A training run is launched using the main `train.py` script and by supplying the config using `-C`
```
python -m train -C configs/train/cfg_twophase_unet.py
```
All config parameters can be overridden using command line arguments. For example, to train on the no-obstacle dataset, using an alternative number of hidden layers for the U-FNO, and using the GPU:
```
python -m train -C configs/train/cfg_twophase_ufno.py --model.hidden_blocks=2 --trainer.device=cuda --dataset.experiment=twophase_no_obstacle
```
***
If this codebase is useful for your work, please consider citing:
```
@inproceedings{
poels2024accelerating,
title={Accelerating Simulation of Two-Phase Flows with Neural {PDE} Surrogates},
author={Yoeri Poels and Koen Minartz and Harshit Bansal and Vlado Menkovski},
booktitle={ICML 2024 AI for Science Workshop},
year={2024},
url={https://openreview.net/forum?id=yIqszw9RUc}
}
```