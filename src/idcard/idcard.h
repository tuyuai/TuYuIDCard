#pragma once
#include "det/detector.h"
#include "rec/recognizer.h"

class IDCardOCR {
 public:
  IDCardOCR(const OrtApi* ort_api, OrtEnv* env)
      : ort_api_(ort_api), env_(env) {}
  virtual ~IDCardOCR();

void InitModel(const std::string& det_model, const std::string& rec_model);
  void ParseHead(const cv::Mat& image,
                 std::vector<std::pair<std::string, std::string>>& infos);
  void ParseEmblem(const cv::Mat& image,
                   std::vector<std::pair<std::string, std::string>>& infos);

 private:
  const OrtApi* ort_api_;
  OrtEnv* env_;
  Detector* detector_;
  Recognizer* recognizer_;
};