# CryoMEM ❄️⫚

### [Project Page (Link DNE](gerdbizi.com/cryoMEM)

_TLDR_: Building on [cryoSPIN](https://shekshaa.github.io/semi-amortized-cryoem/), I incorporate prior estimates on tilt and in-plane rotation to reduce the pose space to just finding the azimuth. This method aims to improve the structural determination of membrane proteins in their native states, can reveal novel interactions that can't be seen when the protein is solubilized. For example, when imaging V-Type ATPase in its native synaptic vesicle membrane, [synaptophysin was discovered to interact with the complex stoichiometrically](https://www.science.org/doi/10.1126/science.adp5577).

## Dependencies
The code is tested on Python `3.9` and Pytorch `1.12` with cuda version `11.3`.
Please run following commands to create a compatible (mini)conda environment called `cryoSPIN`:
```
# Create and activate conda env
conda create -n cryoMEM python=3.9 -y
conda activate cryoMEM

# Install dependencies
bash setup_env.sh
pip install -r requirements.txt
```

[EMAN2](https://cryoem.bcm.edu/cryoem/downloads/view_eman2_versions) is a CLI for aligning predicted maps with their ground truths before computing Fourier Shell Correlation (FSC) and resolution.

## Synthetic Data

To reproduce results, you first need to generate 2D projections based on a given 3D density map. It is recommended that the particle be well-centred.
A density map of V-ATPase is found in the `mrcfiles` folder.
For each dataset, we provide a config file which primarily defines path to density map (`.mrc`), the number projections, and image size (e.g. `128`). 
See the config file for more parameters.
To generate data, run `generate_data.py` with the corresponding config file, e.g. for HSP:
```
python generate_data.py --config ./configs/mrc2star_hsp.yaml
```
As a result, the dataset will get stored locally in `./synthetic_data/hsp/`. 
In this folder, you can find a star file called `data.star` storing the metadata (such as CTF parameters) accompanied with a folder called `Particles` storing particle images into several `.mrcs` files.

Once the synthetic data is ready, you can run the semi-amortized method,
```
python train_semi-amortized.py --config ./configs/train_synth.yaml --save_path path/xyz
```
which will write the reconstruction logs in `path/xyz`. `tensorboard` can be used to view metrics such as reconstruction error.

## Experimental Data
When using real data, a corresponding `cs` file (in the cryoSPARC) format must be provided with the appropriate estimates on the tilt and in-plane rotation. A notebook that uses binarized micrographs indicating vesicles (data taken from [Vesicle Picker](https://github.com/r-karimi/vesicle-picker) will be added to this repo.

The program can be run in the same way as above:
```
python train_semi-amortized.py --config ./configs/train_real.yaml --save_path path/zyx
```
Similarly, the results will be saved in `path/zyx`.
