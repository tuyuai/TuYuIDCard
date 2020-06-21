#include "idcard.h"
#include "det/detector.h"
#include "rec/recognizer.h"


IDCardOCR::~IDCardOCR() {
  if (detector_ != nullptr) {
    delete detector_;
  }
  if (recognizer_ != nullptr) {
    delete recognizer_;
  }
}

void IDCardOCR::InitModel(const std::string& det_model, const std::string& rec_model) {
  detector_ = new Detector(ort_api_, env_);
  recognizer_ = new Recognizer(ort_api_, env_);
  detector_->InitModel(det_model);
  recognizer_->InitModel(rec_model);
}
void IDCardOCR::ParseHead(
    const cv::Mat& image,
    std::vector<std::pair<std::string, std::string>>& infos) {
  std::vector<std::vector<cv::Point2f>> textlines;
  detector_->Predict(image, textlines);
  for (auto textline : textlines) {
    cv::Rect line_rect = cv::boundingRect(textline);
    cv::Mat text_image = image(line_rect);
    std::string res = recognizer_->Predict(text_image);
    std::cout << res << std::endl;
  }
}

void IDCardOCR::ParseEmblem(
    const cv::Mat& image,
    std::vector<std::pair<std::string, std::string>>& infos) {
  std::vector<std::vector<cv::Point2f>> textlines;
  detector_->Predict(image, textlines);
  for (auto textline : textlines) {
    cv::Rect line_rect = cv::boundingRect(textline);
    cv::Mat text_image = image(line_rect);
    std::string res = recognizer_->Predict(text_image);
  }
}
