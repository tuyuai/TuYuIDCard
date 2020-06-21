// Copyright(c) TuYuAI authors.All rights reserved.
// Licensed under the Apache-2.0 License.
//

#include <opencv2/opencv.hpp>
#include "onnxruntime_c_api.h"
#include "recognizer.h"

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
  Recognizer recognizer(g_ort, env);
  recognizer.InitModel(model_path);
  cv::Mat image = cv::imread(image_path);
  std::string res = recognizer.Predict(image);
  std::cout << res << std::endl;
  return 0;
}