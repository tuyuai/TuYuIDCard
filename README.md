# TuYuIDCard
高精度跨平台身份证识别，全C++实现， 支持Windows，Linux， MacOS。

## Prerequst
1. opencv

2. onnxruntime
  在 https://github.com/microsoft/onnxruntime/releases/tag/v1.3.0 下载合适版本的onnxruntime并放在3rdparty目录下，并重命名为onnxruntime。

  
## Build

```shell
git clone https://github.com/tuyuai/TuYuIDCard.git
cd TuYuIDCard && mkdir build && cd build
cmake ..
make -j4
```

## Test

在百度云 链接: https://pan.baidu.com/s/1GwJPgZJiMNzfMcKA3o8ynA 提取码: wq7g 下载模型，然后将模型放在models目录下，进入build目录执行

```shell
./src/idcard_ocr_test ../models/det.onnx ../models/rec.onnx ../images/test.jpg
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).