#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "onnxruntime_c_api.h"

class Detector {
 public:
  Detector(const OrtApi* ort_api, OrtEnv* env)
      : ort_api_(ort_api),
        env_(env),
        session_options_(nullptr),
        session_(nullptr) {}
  ~Detector();

  void GetInputs();
  void GetOutputs();
  void Preprocess(const cv::Mat& image, cv::Mat& out, float& ratio_w,
                  float& ratio_h);
  void InitModel(const std::string& onnx_model_name);
  void Predict(const cv::Mat& image,
               std::vector<std::vector<cv::Point2f>>& bboxes);
  void GetTensorDataAndShape(OrtValue* input_map, float** array,
                             std::vector<int64_t>& node_dims);

  cv::Mat ShowTextLines(const cv::Mat& input,
                        std::vector<std::vector<cv::Point2f>>& bboxes);

 private:
  const OrtApi* ort_api_;
  OrtEnv* env_;
  OrtSessionOptions* session_options_;
  OrtSession* session_;

  std::vector<const char*> input_node_names_;
  std::vector<int64_t> input_node_dims_;
};
