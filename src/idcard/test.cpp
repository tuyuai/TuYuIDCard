#include <opencv2/opencv.hpp>
#include "idcard.h"
#include "onnxruntime_c_api.h"

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "idcard_rec_test det_model_path rec_model_path image_path"
              << std::endl;
    return 0;
  }
  std::string det_model_path = argv[1];
  std::string rec_model_path = argv[2];
  std::string image_path = argv[3];
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env;
  g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "idcard", &env);
  IDCardOCR idcard(g_ort, env);
  idcard.InitModel(det_model_path, rec_model_path);
  cv::Mat image = cv::imread(image_path);
  std::vector<std::pair<std::string, std::string>> infos;
  idcard.ParseHead(image, infos);

  return 0;
}