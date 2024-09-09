#!/bin/bash

npm install
cd shaders/
python3 embed_shaders.py ./glslc.exe ./tint.exe
cd ..
npm run build
python3 -m pip install gdown numpy matplotlib
. $HOME/.profile
gdown 1iAN-LucPq6nUAh74I1BIXa24KaXo650k
gdown 1t98uqIjGB99k3Xso8R1EQL4fgefHlKBR
gdown 1YTBFATCaK1ApFpcefEuAj5iQTPm998pU
mkdir dist/models/bcmc-data -p
mv chameleon_1024x1024x1080_uint16.raw.crate2.zfp dist/models/bcmc-data
mv magnetic_reconnection_512x512x512_float32.raw.crate4.zfp dist/models/bcmc-data
mv miranda_1024x1024x1024_float32.raw.crate4.zfp dist/models/bcmc-data
cp ml-models/noof640-ultraminiv12.onnx dist/
cp ml-models/noof1280-ultraminiv12.onnx dist/
cp ml-models/noof1920-ultraminiv12.onnx dist/
cd dist/
python3 -m http.server
