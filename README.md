## Installation

Clone the repo locally

`git clone https://github.com/saptarshineilsinhaigd/spectralGaussian.git`

`cd spectralGaussian`


``` 
conda create -n spectral_gaussian python=3.8 -y
conda activate spectral_gaussian 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn 

mkdir thirdparty
cd thirdparty
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
```

Note did not test this long time. I used my conda and could be some dependencies are missing. If there are dependency issues get back to me.

## How to run

### Dataset

If you want to prepare masks on your own dataset

```
cd Tracking-Anything-with-DEVA
pip install -e .
bash scripts/download_models.sh     # Download the pretrained models

git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO

cd ../..
```

Added example cross-spectral data for penguins.
We will release the entire dataset after clean-up.


### Scripts

#### Training

`chmod +x train.sh`

`./train.sh`

#### Render novel views

`chmod +x render.sh`

`./render.sh`