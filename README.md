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
For either install, it is necessary to be on a device with both npm and python installed. Code was created using Windows Subsystem for Linux, but is compatible with Windows, Mac or Linux. 

### Automatic Install
After cloning the repo, install needed dependencies and start serving the application with
```
./run_server.sh
```
From here, the application will be served at localhost:8000.

### Manual Install
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

Then move the files in the ml-models/ folder into the built dist/ folder. 

You can then serve the application from the dist/ folder using 
```
python -m http.server
```
Which will default to serving the application at localhost:8000.

### Running Benchmarks
Once the application is hosted, visit 'localhost:8000/#autobenchmark=0' to begin benchmarks. This will automatically run 27 benchmarks including the Plasma, Chameleon, and Miranda datasets at 360p, 720p, and 1080p, and download .json benchmark files to your default download location.

### Converting Benchmarks to Data Figure
Once the autobenchmark is complete, move all downloaded .json files to the benchmarks/ folder in this repo. Run
```
python plot_figure6.py
```
and files labeled "ResultsAt85%Complete.png" and "ResultsAt100%Complete.png" will be created in the folder, matching Figure 6 in the TVCG paper.

## Model Training
Another repo is provided containing all the model training code [here](https://github.com/ldyken53/TVCG-progiso-training). This repo includes checkpoints for our pretrained model and example data for training new models. Unlike this repo, an NVIDIA GPU with CUDA support is required for model training code. 
