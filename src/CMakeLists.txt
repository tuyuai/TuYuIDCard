include_directories(${PROJECT_SOURCE_DIR}/src)

add_library(idcard_det SHARED det/detector.cpp det/clipper/clipper.cpp)
target_link_libraries(idcard_det ${OpenCV_LIBS} onnxruntime)

add_executable(idcard_det_test det/test.cpp)
target_link_libraries(idcard_det_test idcard_det ${OpenCV_LIBS} onnxruntime)

add_library(idcard_rec SHARED rec/recognizer.cpp rec/decode.cpp)
target_link_libraries(idcard_rec ${OpenCV_LIBS} onnxruntime)

add_executable(idcard_rec_test rec/test.cpp)
target_link_libraries(idcard_rec_test idcard_rec ${OpenCV_LIBS} onnxruntime)

add_library(idcard_ocr SHARED idcard/idcard.cpp)
target_link_libraries(idcard_ocr ${OpenCV_LIBS} onnxruntime idcard_det idcard_rec)

add_executable(idcard_ocr_test idcard/test.cpp)
target_link_libraries(idcard_ocr_test idcard_ocr ${OpenCV_LIBS} onnxruntime)