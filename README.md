# WebGPU Progressive Raycasting with ML Infill
This repo expands https://github.com/Twinklebear/webgpu-prog-iso with multiple optimizations including first pass speculation, 
starting with speculation using larger framebuffers, and ML infill using onnxruntime-web.

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
