include_directories(${CMAKE_CURRENT_DIR})
include_directories(${CMAKE_CURRENT_DIR}/../)
add_library(idcard_ocr SHARED idcard.cpp)
target_link_libraries(idcard_ocr ${OpenCV_LIBS} onnxruntime)

add_executable(idcard_ocr_test test.cpp)
target_link_libraries(idcard_ocr_test idcard_ocr ${OpenCV_LIBS} onnxruntime)
