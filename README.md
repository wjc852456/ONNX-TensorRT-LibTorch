
# Deploying Deep Learning Models with ONNX+TensorRT+LibTorch
This project is based on https://github.com/dusty-nv/jetson-inference#api-reference. 

With pretrained model exported as .onnx model, runtime engines are created and deployed along with **TensorRT** and **LibTorch** on NVIDIA Geforce or embedded Jetson platforms.

### Setup
modify build.cmake to specify the path of TRT_INCLUDE_DIR and TORCH_LIB_PATH (if TensorRT was pre-installed like in Jetson, TRT_INCLUDE_DIR should not be specified), then run the commands below:

$ mkdir build

$ cp build.cmake build

$ cd build

$ cmake .. -DCMAKE_BUILD_TYPE=Release

$ make 


### Object Detection

| Network         |  runtime precision   | Object classes       | resolution | Jetson NX |
| ----------------|--------------------  |----------------------|------------|-----------|
| YOLOV5s         | FP16   | 4 | 1920*1080 |100FPS |
