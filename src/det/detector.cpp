// Copyright(c) TuYuAI authors.All rights reserved.
// Licensed under the Apache-2.0 License.
//

#include "detector.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include "det/lanms.hpp"
#include "spdlog/spdlog.h"
using namespace lanms;

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

Detector::~Detector() {
  if (session_ != nullptr) {
    ort_api_->ReleaseSession(session_);
  }
  if (session_options_ != nullptr) {
    ort_api_->ReleaseSessionOptions(session_options_);
  }
}

void Detector::InitModel(const std::string& model_path) {
  ORT_ABORT_ON_ERROR(ort_api_->CreateSessionOptions(&session_options_));
  ort_api_->SetIntraOpNumThreads(session_options_, 1);
  ort_api_->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL);

  ORT_ABORT_ON_ERROR(ort_api_->CreateSession(env_, model_path.c_str(),
                                             session_options_, &session_));
}

void Detector::GetInputs() {
  size_t num_input_nodes;
  OrtStatus* status;
  OrtAllocator* allocator;
  ORT_ABORT_ON_ERROR(ort_api_->GetAllocatorWithDefaultOptions(&allocator));
  ort_api_->SessionGetInputCount(session_, &num_input_nodes);
  input_node_names_.resize(num_input_nodes);
  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name;
    status = ort_api_->SessionGetInputName(session_, i, allocator, &input_name);
    printf("Input %zu : name=%s\n", i, input_name);
    input_node_names_[i] = input_name;

    // print input node types
    OrtTypeInfo* typeinfo;
    status = ort_api_->SessionGetInputTypeInfo(session_, i, &typeinfo);
    const OrtTensorTypeAndShapeInfo* tensor_info;
    ORT_ABORT_ON_ERROR(
        ort_api_->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    ONNXTensorElementDataType type;
    ORT_ABORT_ON_ERROR(ort_api_->GetTensorElementType(tensor_info, &type));
    printf("Input %zu : type=%d\n", i, type);

    // print input shapes/dims
    size_t num_dims;
    ORT_ABORT_ON_ERROR(ort_api_->GetDimensionsCount(tensor_info, &num_dims));
    printf("Input %zu : num_dims=%zu\n", i, num_dims);
    input_node_dims_.resize(num_dims);
    ort_api_->GetDimensions(tensor_info, (int64_t*)input_node_dims_.data(),
                            num_dims);
    for (size_t j = 0; j < num_dims; j++)
      printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims_[j]);

    ort_api_->ReleaseTypeInfo(typeinfo);
  }
}

void Detector::GetTensorDataAndShape(OrtValue* input_map, float** array,
                                     std::vector<int64_t>& node_dims) {
  ORT_ABORT_ON_ERROR(ort_api_->GetTensorMutableData(input_map, (void**)array));

  OrtTensorTypeAndShapeInfo* tensor_info;
  ORT_ABORT_ON_ERROR(ort_api_->GetTensorTypeAndShape(input_map, &tensor_info));

  size_t num_dims;
  ORT_ABORT_ON_ERROR(ort_api_->GetDimensionsCount(tensor_info, &num_dims));
  node_dims.resize(num_dims);
  ORT_ABORT_ON_ERROR(ort_api_->GetDimensions(
      tensor_info, (int64_t*)node_dims.data(), num_dims));
  ort_api_->ReleaseTensorTypeAndShapeInfo(tensor_info);
}

void Detector::Preprocess(const cv::Mat& input_image, cv::Mat& out_image,
                          float& ratio_w, float& ratio_h) {
  cv::Mat image;
  cv::cvtColor(input_image, image, cv::COLOR_BGR2RGB);
  int input_width = image.cols;
  int input_height = image.rows;
  int param_w = 602;
  int param_h = 378;

  int param_max_side_length = 1200;
  float ratio = 1.0;
  if (std::max(input_height, input_width) > param_max_side_length) {
    ratio = float(param_max_side_length) / input_height;
    if (input_height < input_width) {
      ratio = float(param_max_side_length) / input_width;
    }
  }

  int new_h = int(input_height * ratio);
  int new_w = int(input_width * ratio);

  float param_h_ratio = (float)param_h / float(new_h);
  float param_w_ratio = (float)param_w / float(new_w);

  ratio = std::min(param_w_ratio, param_h_ratio);
  new_h = int(new_h * ratio);
  new_w = int(new_w * ratio);

  if (new_h % 32 != 0) {
    new_h = int(new_h / 32) * 32;
  }
  if (new_w % 32 != 0) {
    new_w = int(new_w / 32) * 32;
  }

  cv::Mat resize_image;
  cv::resize(image, resize_image, cv::Size(new_w, new_h));
  resize_image.convertTo(out_image, CV_32FC3);
  SPDLOG_INFO("image resize from [{} {}] -> [{} {}]", input_width, input_height,
              new_w, new_h);

  ratio_w = new_w / float(input_width);
  ratio_h = new_h / float(input_height);
}

std::vector<float> RestoreRBox(float* geo_array, float* score_array, int height,
                               int width) {
  std::vector<float> quad_data;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int idx = i * width + j;
      if (score_array[idx] > 0.8) {
        int x = j * 4;
        int y = i * 4;
        float d0 = geo_array[i * width * 5 + j * 5 + 0];
        float d1 = geo_array[i * width * 5 + j * 5 + 1];
        float d2 = geo_array[i * width * 5 + j * 5 + 2];
        float d3 = geo_array[i * width * 5 + j * 5 + 3];
        float angle = geo_array[i * width * 5 + j * 5 + 4];
        if (angle >= 0) {
          float a0 = 0;
          float a1 = -d0 - d2;
          float a2 = d1 + d3;
          float a3 = -d0 - d2;
          float a4 = d1 + d3;
          float a5 = 0;
          float a6 = 0;
          float a7 = 0;
          float a8 = d3;
          float a9 = -d2;

          float rotate_x0 = std::cos(angle);
          float rotate_x1 = std::sin(angle);

          float rotate_y0 = -std::sin(angle);
          float rotate_y1 = std::cos(angle);

          float bx0 = rotate_x0 * a0 + rotate_x1 * a1;
          float bx1 = rotate_x0 * a2 + rotate_x1 * a3;
          float bx2 = rotate_x0 * a4 + rotate_x1 * a5;
          float bx3 = rotate_x0 * a6 + rotate_x1 * a7;
          float bx4 = rotate_x0 * a8 + rotate_x1 * a9;

          float by0 = rotate_y0 * a0 + rotate_y1 * a1;
          float by1 = rotate_y0 * a2 + rotate_y1 * a3;
          float by2 = rotate_y0 * a4 + rotate_y1 * a5;
          float by3 = rotate_y0 * a6 + rotate_y1 * a7;
          float by4 = rotate_y0 * a8 + rotate_y1 * a9;

          float org_x = x - bx4;
          float org_y = y - by4;
          float new_px0 = bx0 + org_x;
          float new_py0 = by0 + org_y;
          float new_px1 = bx1 + org_x;
          float new_py1 = by1 + org_y;
          float new_px2 = bx2 + org_x;
          float new_py2 = by2 + org_y;
          float new_px3 = bx3 + org_x;
          float new_py3 = by3 + org_y;

          quad_data.push_back(new_px0);
          quad_data.push_back(new_py0);
          quad_data.push_back(new_px1);
          quad_data.push_back(new_py1);
          quad_data.push_back(new_px2);
          quad_data.push_back(new_py2);
          quad_data.push_back(new_px3);
          quad_data.push_back(new_py3);
          quad_data.push_back(score_array[idx]);
        } else {
          float a0 = -d1 - d3;
          float a1 = -d0 - d2;
          float a2 = 0;
          float a3 = -d0 - d2;
          float a4 = 0;
          float a5 = 0;
          float a6 = -d1 - d3;
          float a7 = 0;
          float a8 = -d1;
          float a9 = -d2;

          float rotate_x0 = std::cos(-angle);
          float rotate_x1 = -std::sin(-angle);

          float rotate_y0 = std::sin(-angle);
          float rotate_y1 = std::cos(-angle);

          float bx0 = rotate_x0 * a0 + rotate_x1 * a1;
          float bx1 = rotate_x0 * a2 + rotate_x1 * a3;
          float bx2 = rotate_x0 * a4 + rotate_x1 * a5;
          float bx3 = rotate_x0 * a6 + rotate_x1 * a7;
          float bx4 = rotate_x0 * a8 + rotate_x1 * a9;

          float by0 = rotate_y0 * a0 + rotate_y1 * a1;
          float by1 = rotate_y0 * a2 + rotate_y1 * a3;
          float by2 = rotate_y0 * a4 + rotate_y1 * a5;
          float by3 = rotate_y0 * a6 + rotate_y1 * a7;
          float by4 = rotate_y0 * a8 + rotate_y1 * a9;

          float org_x = x - bx4;
          float org_y = y - by4;
          float new_px0 = bx0 + org_x;
          float new_py0 = by0 + org_y;
          float new_px1 = bx1 + org_x;
          float new_py1 = by1 + org_y;
          float new_px2 = bx2 + org_x;
          float new_py2 = by2 + org_y;
          float new_px3 = bx3 + org_x;
          float new_py3 = by3 + org_y;

          quad_data.push_back(new_px0);
          quad_data.push_back(new_py0);
          quad_data.push_back(new_px1);
          quad_data.push_back(new_py1);
          quad_data.push_back(new_px2);
          quad_data.push_back(new_py2);
          quad_data.push_back(new_px3);
          quad_data.push_back(new_py3);
          quad_data.push_back(score_array[idx]);
        }
      }
    }
  }

  return quad_data;
}

void Detector::Predict(const cv::Mat& image,
                       std::vector<std::vector<cv::Point2f>>& textlines) {
  cv::Mat out_image;
  float ratio_h;
  float ratio_w;
  Preprocess(image, out_image, ratio_w, ratio_h);
  SPDLOG_DEBUG("image h={} w={} resize h={} w={} ratio h={} ratio w = {}",
               image.rows, image.cols, out_image.rows, out_image.cols, ratio_h,
               ratio_w);
  int image_width = out_image.cols;
  int image_height = out_image.rows;
  int image_channels = out_image.channels();

  std::vector<int64_t> input_node_dims = {1, image_height, image_width,
                                          image_channels};
  size_t input_tensor_size = image_width * image_height * image_channels;

  // create input tensor object from data values
  OrtMemoryInfo* allocator_info;
  ORT_ABORT_ON_ERROR(ort_api_->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault, &allocator_info));
  OrtValue* input_tensor = NULL;
  ORT_ABORT_ON_ERROR(ort_api_->CreateTensorWithDataAsOrtValue(
      allocator_info, out_image.data, input_tensor_size * sizeof(float),
      input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &input_tensor));
  int is_tensor;
  ORT_ABORT_ON_ERROR(ort_api_->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);

  OrtStatus* status;
  // const char* input_names[] = {"input_images:0"};
  // const char* output_names[] = {"feature_fusion/concat_3:0",
  //                               "feature_fusion/Conv_7/Sigmoid:0"};
  const char* input_names[] = {"input"};
  const char* output_names[] = {"geo_map", "score_map"};

  OrtValue* output_tensor[2];
  output_tensor[0] = NULL;
  output_tensor[1] = NULL;
  SPDLOG_INFO("East run begin");
  ORT_ABORT_ON_ERROR(ort_api_->Run(this->session_, NULL, input_names,
                                   (const OrtValue* const*)&input_tensor, 1,
                                   output_names, 2, output_tensor));
  SPDLOG_INFO("East run end");
  OrtValue* geo_map = output_tensor[0];
  OrtValue* score_map = output_tensor[1];

  std::vector<int64_t> geo_shape;
  float* geo_array;
  GetTensorDataAndShape(geo_map, &geo_array, geo_shape);
  std::vector<int64_t> score_shape;
  float* score_array;
  GetTensorDataAndShape(score_map, &score_array, score_shape);

  int batch = score_shape[0];
  int height = score_shape[1];
  int width = score_shape[2];
  int geo_count = geo_shape[3];
  int score_count = score_shape[3];

  std::vector<float> quad_data =
      RestoreRBox(geo_array, score_array, height, width);

  for (int i = 0; i < quad_data.size() / 9; i++) {
    for (int j = 0; j < 8; j++) {
      quad_data[i * 9 + j] *= 10000.0f;
    }
  }
  std::vector<Polygon> polys =
      merge_quadrangle_n9(quad_data.data(), quad_data.size() / 9, 0.2);
  std::vector<std::vector<float>> boxes = polys2floats_new(polys);

  for (int i = 0; i < boxes.size(); i++) {
    auto box = boxes[i];
    std::vector<cv::Point2f> line_item;
    for (int i = 0; i < 4; i++) {
      line_item.push_back(
          cv::Point2f(box[2 * i + 0] / ratio_w, box[2 * i + 1] / ratio_h));
    }
    textlines.push_back(line_item);
  }

  ort_api_->ReleaseValue(geo_map);
  ort_api_->ReleaseValue(score_map);
  ort_api_->ReleaseValue(input_tensor);
  ort_api_->ReleaseMemoryInfo(allocator_info);
  return;
}

cv::Mat Detector::ShowTextLines(
    const cv::Mat& input, std::vector<std::vector<cv::Point2f>>& textlines) {
  cv::Mat image = input.clone();
  for (int i = 0; i < textlines.size(); i++) {
    std::vector<cv::Point2f>& pts = textlines[i];
    cv::line(image, pts[0], pts[1], cv::Scalar(0, 255, 0), 2);
    cv::line(image, pts[1], pts[2], cv::Scalar(0, 255, 0), 2);
    cv::line(image, pts[2], pts[3], cv::Scalar(0, 255, 0), 2);
    cv::line(image, pts[3], pts[0], cv::Scalar(0, 255, 0), 2);
  }

  return image;
}