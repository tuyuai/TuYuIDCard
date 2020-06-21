// Copyright(c) TuYuAI authors.All rights reserved.
// Licensed under the Apache-2.0 License.
//

#include <opencv2/opencv.hpp>
#include "det/detector.h"
#include "onnxruntime_c_api.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "idcard_rec_test model_path image_path" << std::endl;
    return 0;
  }
  std::string model_path = argv[1];
  std::string image_path = argv[2];
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env;
  g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "idcard", &env);
  Detector detector(g_ort, env);
  detector.InitModel(model_path);
  std::vector<std::vector<cv::Point2f> > textlines;
  cv::Mat image = cv::imread(image_path);
  detector.Predict(image, textlines);
  cv::Mat show_image = detector.ShowTextLines(image, textlines);
  cv::imwrite("show.png", show_image);
  return 0;
}