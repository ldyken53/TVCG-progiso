#!/bin/bash

npm install
cd shaders/
python embed_shaders.py ./glslc.exe ./tint.exe
cd ..
npm run build
cp ml-models/noof640-ultraminiv12.onnx dist/
cp ml-models/noof1280-ultraminiv12.onnx dist/
cp ml-models/noof1920-ultraminiv12.onnx dist/
cd dist/
export QT_QPA_PLATFORM=offscreen
python -m http.server