
# Deploying Deep Learning Models with ONNX+TensorRT+LibTorch
This project is based on https://github.com/dusty-nv/jetson-inference#api-reference. 

With pretrained model exported as .onnx model, runtime engines are created and deployed along with **TensorRT** and **LibTorch** on NVIDIA Geforce or embedded Jetson platforms.

## Pre-Trained Models


#### Object Detection

| Network         |  runtime precision   | Object classes       | resolution | Jetson NX |
| ----------------|--------------------  |----------------------|------------|-----------|
| YOLOV5s         | FP16   | 4 | 1920*1080 |100FPS |
