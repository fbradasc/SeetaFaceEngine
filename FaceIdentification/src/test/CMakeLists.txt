aux_source_directory (. SRC_LIST)

message(${SRC_LIST})

# add external libraries
find_package(OpenCV REQUIRED)

include_directories(/home/fbradasc/sorgenti/SeetaFaceEngine/FaceAlignment/include/)
include_directories(/home/fbradasc/sorgenti/SeetaFaceEngine/FaceDetection/include/)

link_directories(/home/fbradasc/sorgenti/SeetaFaceEngine/FaceAlignment/build/)
link_directories(/home/fbradasc/sorgenti/SeetaFaceEngine/FaceDetection/build/)


enable_testing ()
foreach (f ${SRC_LIST})
  string(REGEX REPLACE "[.]cpp" ".bin" BIN ${f})
  add_executable(${BIN} ${f})
  target_link_libraries(${BIN} viplnet ${OpenCV_LIBS} seeta_facedet_lib seeta_fa_lib)
endforeach ()
