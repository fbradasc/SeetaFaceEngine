aux_source_directory (. SRC_LIST)

message(${SRC_LIST})

# add external libraries
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/../FaceAlignment/include/)
include_directories(${PROJECT_SOURCE_DIR}/../FaceDetection/include/)

link_directories(${PROJECT_SOURCE_DIR}/../FaceAlignment/build/)
link_directories(${PROJECT_SOURCE_DIR}/../FaceDetection/build/)

enable_testing ()
foreach (f ${SRC_LIST})
  string(REGEX REPLACE "[.]cpp" ".bin" BIN ${f})
  add_executable(${BIN} ${f})
  target_link_libraries(${BIN} viplnet ${OpenCV_LIBS} seeta_facedet_lib seeta_fa_lib)
endforeach ()
