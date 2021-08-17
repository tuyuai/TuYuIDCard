cmake_minimum_required(VERSION 3.5)
project(TuYuIDCard)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${PROJECT_SOURCE_DIR}/third_party)

if(MSVC)
	set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/third_party/win64/opencv/x64/vc15/lib)
	find_package(OpenCV REQUIRED)
	include_directories(${OpenCV_INCLUDE_DIR})
	include_directories(${PROJECT_SOURCE_DIR}/third_party/win64/onnxruntime-win-x64-1.8.0/include)
	link_directories(${PROJECT_SOURCE_DIR}/third_party/win64/onnxruntime-win-x64-1.8.0/lib)
else(MSVC)
	find_package(OpenCV REQUIRED)
	include_directories(${OpenCV_INCLUDE_DIR})
	include_directories(${PROJECT_SOURCE_DIR}/third_party/onnxruntime/include)
	link_directories(${PROJECT_SOURCE_DIR}/third_party/onnxruntime/lib)
endif()

add_subdirectory(src)