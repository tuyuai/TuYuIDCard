# TuYuIDCard
TuYuIDCard高精度跨平台身份证识别，全C++实现， 支持Windows，Linux， MacOS。

在线身份证OCR demo [**TuYuIDCard**](http://49.235.99.66/)

## 下载代码和依赖
本项目依赖opencv-4.5.2， onnxruntime-1.8.0,  windows下要求VisualStudio2019，linux 下GCC>6.0， cmake > 3.5。

下载代码

`git clone https://github.com/tuyuai/TuYuIDCard.git`

下载模型和依赖

在百度网盘下载  链接：https://pan.baidu.com/s/1V-xcXYJptHy_ud51P_aCGw  提取码：ygho ， 并放入TuYuIDCard目录下。



## Build

```shell
cd TuYuIDCard && mkdir build && cd build
cmake ..
make -j4
```

## Test

在build目录下执行

```shell
./src/idcard_ocr_test ../models/det.onnx ../models/rec.onnx ../images/test.jpg
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).