#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "decode.h"
#include "onnxruntime_c_api.h"

class Recognizer {
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

