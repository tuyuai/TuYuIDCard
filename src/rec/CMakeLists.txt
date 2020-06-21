include_directories(${CMAKE_CURRENT_DIR})
add_library(idcard_rec SHARED recognizer.cpp decode.cpp)
target_link_libraries(idcard_rec ${OpenCV_LIBS} onnxruntime)

add_executable(idcard_rec_test test.cpp)
target_link_libraries(idcard_rec_test idcard_rec ${OpenCV_LIBS} onnxruntime)
