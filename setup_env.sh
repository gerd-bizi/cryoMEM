#!/bin/bash

echo "Installing Python packages for cryoSPIN..."

# Install PyTorch with CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install additional libraries
pip install scipy==1.9.3 scikit-image==0.22.0 tqdm PyYAML matplotlib kornia \
    notebook tensorboard numpy==1.23.4 starfile==0.4.5 mrcfile

# Install PyTorch3D from stable branch
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo "All dependencies installed successfully!"
