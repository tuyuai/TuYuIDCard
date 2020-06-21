# TuYuIDCard
高精度跨平台身份证识别，全C++实现， 支持Windows，Linux， MacOS。

## Prerequst
1. opencv

2. onnxruntime
  在 https://github.com/microsoft/onnxruntime/releases/tag/v1.3.0 下载合适版本的onnxruntime并放在3rdparty目录下，出现命名为onnxruntime。

  -3rdparty

  ​	--onnxruntime

  ​	
## Build

```shell
git clone https://github.com/tuyuai/TuYuIDCard.git
cd TuYuIDCard && mkdir build && cd build
cmake ..
make -j4
```



## License


