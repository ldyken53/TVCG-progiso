# WebGPU Isosurface Visualization
This repo holds the code for the TVCG paper, ["Interactive Isosurface Visualization in Memory Constrained Environments Using Deep Learning and Speculative Raycasting"](https://ieeexplore.ieee.org/document/10577555) by Landon Dyken, Will Usher, and Sidharth Kumar. This work expands the algorithm of ["Speculative Progressive Raycasting for Memory Constrained Isosurface Visualization of Massive Volumes"](https://github.com/Twinklebear/webgpu-prog-iso) (LDAV 2023 Best Paper) by using a pretrained image reconstruction network to infer perceptual approximates from intermediate output, along with optimizing the speculative raycasting using first pass speculation and larger computational buffers to increase speculation counts in early passes.

## Demo 
There is an interactive demo for several datasets online:
- [Magnetic Reconnection (Plasma)](https://ldyken53.github.io/TVCG-progiso/#dataset=magnetic) (512^3)
- [Chameleon](https://ldyken53.github.io/TVCG-progiso/#dataset=chameleon) (1024x1024x1080)
- [Miranda](https://ldyken53.github.io/TVCG-progiso/#dataset=miranda) (1024^3)
- [Richtmyer Meshkov](https://ldyken53.github.io/TVCG-progiso/#dataset=richtmyer_meshkov) (2048x2048x1920)

Note that due to initially loading the datasets, it will take some time for the rendering to appear when visiting the pages for the first time.

All datasets are available on the [Open SciVis Datasets page](https://klacansky.com/open-scivis-datasets/). 

## Getting Started

After cloning the repo run

```
npm install
```

Then navigate to the shaders/ folder and run
```
python embed_shaders.py ./glslc.exe ./tint.exe
```

Then back to the top folder run 
```
npm run build
```

Then move the files in "ml-models" into the built dist/ folder. Then create a folder
"models/bcmc-data" in dist/ and populate with the zfp compressed datasets. 

You can then serve the application from the dist/ folder using 
```
python -m http.server
```

## Model Training
Another repo is provided containing all the model training code [here](https://github.com/ldyken53/TVCG-progiso-training). This repo includes checkpoints for our pretrained model and example data for training new models. Unlike this repo, an NVIDIA GPU with CUDA support is required for model training code. 
