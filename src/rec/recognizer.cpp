// Copyright(c) TuYuAI authors.All rights reserved.
// Licensed under the Apache-2.0 License.
// 

#include "recognizer.h"
#include <onnxruntime_c_api.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include "spdlog/spdlog.h"

#define ORT_ABORT_ON_ERROR(expr)                                \
  do {                                                          \
    OrtStatus* onnx_status = (expr);                            \
    if (onnx_status != NULL) {                                  \
      const char* msg = ort_api_->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                             \
      ort_api_->ReleaseStatus(onnx_status);                     \
      abort();                                                  \
    }                                                           \
  } while (0);

Recognizer::~Recognizer() {
  if (session_ != nullptr) {
    ort_api_->ReleaseSession(session_);
  }
  if (session_options_ != nullptr) {
    ort_api_->ReleaseSessionOptions(session_options_);
  }
}

void Recognizer::InitModel(const std::string& model_path) {
  ORT_ABORT_ON_ERROR(ort_api_->CreateSessionOptions(&session_options_));
  ort_api_->SetIntraOpNumThreads(session_options_, 1);
  ort_api_->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL);
  ORT_ABORT_ON_ERROR(ort_api_->CreateSession(env_, model_path.c_str(),
                                             session_options_, &session_));
}

std::string Recognizer::Predict(const cv::Mat& image) {
  auto start_time = std::chrono::high_resolution_clock::now();

  cv::Mat out;
  Preprocess(image, out);

  int image_width = out.cols;
  int image_height = out.rows;
  int image_channels = out.channels();

  std::vector<int64_t> input_node_dims = {1, image_channels, image_height,
                                          image_width};
  size_t input_tensor_size = image_width * image_height * image_channels;
  std::vector<float> input_tensor_values(input_tensor_size);

  float* input_data = input_tensor_values.data();

  for (int h = 0; h < image_height; ++h) {
    for (int w = 0; w < image_width; ++w) {
      int idx0 = h * image_width + w;
      int idx1 = image_height * image_width + idx0;
      int idx2 = 2 * image_height * image_width + idx0;
      cv::Vec3f d = out.at<cv::Vec3f>(h, w);
      input_tensor_values[idx0] = d[0];
      input_tensor_values[idx1] = d[1];
      input_tensor_values[idx2] = d[2];
    }
  }

  // create input tensor object from data values
  OrtMemoryInfo* allocator_info;
  ORT_ABORT_ON_ERROR(ort_api_->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault, &allocator_info));
  OrtValue* input_tensor = NULL;
  ORT_ABORT_ON_ERROR(ort_api_->CreateTensorWithDataAsOrtValue(
      allocator_info, input_tensor_values.data(),
      input_tensor_size * sizeof(float), input_node_dims.data(), 4,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
  int is_tensor;
  ORT_ABORT_ON_ERROR(ort_api_->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);

  OrtStatus* status;
  const char* input_names[] = {"image"};
  const char* output_names[] = {"output"};

  OrtValue* output_tensor = NULL;
  ORT_ABORT_ON_ERROR(ort_api_->Run(this->session_, NULL, input_names,
                                   (const OrtValue* const*)&input_tensor, 1,
                                   output_names, 1, &output_tensor));
  assert(output_tensor != NULL);
  ORT_ABORT_ON_ERROR(ort_api_->IsTensor(output_tensor, &is_tensor));
  assert(is_tensor);

  float* out_array;
  ORT_ABORT_ON_ERROR(
      ort_api_->GetTensorMutableData(output_tensor, (void**)&out_array));

  OrtTensorTypeAndShapeInfo* output_tensor_info;
  ORT_ABORT_ON_ERROR(
      ort_api_->GetTensorTypeAndShape(output_tensor, &output_tensor_info));

  size_t out_num_dims;
  ORT_ABORT_ON_ERROR(
      ort_api_->GetDimensionsCount(output_tensor_info, &out_num_dims));
  std::vector<int64_t> output_node_dims;
  output_node_dims.resize(out_num_dims);
  ORT_ABORT_ON_ERROR(ort_api_->GetDimensions(
      output_tensor_info, (int64_t*)output_node_dims.data(), out_num_dims));

  int64_t T = output_node_dims[0];
  int64_t N = output_node_dims[1];
  int64_t C = output_node_dims[2];
  SPDLOG_DEBUG("T = {}, N= {}, C={}\n", T, N, C);
  std::vector<int> preds;
  for (int t = 0; t < T; t++) {
    int idx = 0;
    float max_value = -10000000000.0f;
    for (int c = 0; c < C; c++) {
      if (out_array[t * C + c] > max_value) {
        max_value = out_array[t * C + c];
        idx = c;
      }
    }
    preds.emplace_back(idx);
    SPDLOG_DEBUG("preds is {}\n", idx);
  }

  std::vector<int> result = GreedyDecode(preds);

  ort_api_->ReleaseTensorTypeAndShapeInfo(output_tensor_info);
  ort_api_->ReleaseValue(output_tensor);
  ort_api_->ReleaseValue(input_tensor);
  ort_api_->ReleaseMemoryInfo(allocator_info);

  std::string ret_result;
  for (int i = 0; i < result.size(); i++) {
    int idx = result[i];
    ret_result += alphabets[idx-1];
  }

  return ret_result;
}

void Recognizer::Preprocess(const cv::Mat& image, cv::Mat& out) {
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  int image_width = image.cols;
  int image_height = image.rows;
  int param_w = 200;
  int param_h = 32;
  float ratio = float(param_w) / float(param_h);
  float h_major_ratio = float(image_height) / float(param_h);
  int new_h = int(image_height / h_major_ratio);
  int new_w = int(image_width / h_major_ratio);
  SPDLOG_DEBUG("new_h = {}, new_w =  {}", new_h, new_w);
  cv::Mat resize_image;
  cv::resize(image, resize_image, cv::Size(new_w, new_h));
  if (((float)image_width / image_height) < ratio) {
    int top = (param_h - new_h) / 2;
    int left = (param_w - new_w) / 2;
    cv::Mat pad_image(cv::Size(param_w, param_h), image.type());
    pad_image.setTo(cv::Scalar(255, 255, 255));
    cv::Mat roi_image = pad_image(cv::Rect(left, top, new_w, new_h));
    resize_image.copyTo(roi_image);
    cv::resize(pad_image, resize_image, cv::Size(param_w, param_h));
  }
  cv::Mat out_float;
  resize_image.convertTo(out_float, CV_32FC3);
  out = out_float / 255.0f;
  out = (out - 0.5) / 0.5;
}
