// Copyright(c) TuYuAI authors.All rights reserved.
// Licensed under the Apache-2.0 License.
//

#ifndef RECOGNIZER_H_
#define RECOGNIZER_H_
#include "common/common.h"
#include <opencv2/opencv.hpp>
#include <string>
#include "decode.h"
#include "onnxruntime_c_api.h"

class TUYUIDCARD_API Recognizer {
 public:
  Recognizer(const OrtApi* ort_api, OrtEnv* env)
      : ort_api_(ort_api),
        env_(env),
        session_(nullptr),
        session_options_(nullptr) {}
  ~Recognizer();
  std::string Predict(const cv::Mat& image);

  void Preprocess(const cv::Mat& image, cv::Mat& out);
  void InitModel(const std::string& onnx_model_name);

 private:
  const OrtApi* ort_api_;
  OrtEnv* env_;
  OrtSessionOptions* session_options_;
  OrtSession* session_;
};

#endif 