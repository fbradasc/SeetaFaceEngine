/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face alignment, the
 * face alignment method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment, 
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "face_detection.h"
#include "face_alignment.h"

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
#endif

using namespace cv;

int main(int argc, char** argv)
{
  // Initialize face detection model
  seeta::FaceDetection detector((MODEL_DIR + "seeta_fd_frontal_v1.0.bin").c_str());
  detector.SetMinFaceSize(40);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);

  // Initialize face alignment model 
  seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());

  //load image
  cv::Mat img_color = cv::imread((DATA_DIR + "image_0001.png").c_str(), 1);
  int pts_num = 5;

  cv::Mat img = cv::imread((DATA_DIR + "image_0001.png").c_str(), cv::IMREAD_UNCHANGED);
  cv::Mat img_gray;

  if (img.channels() != 1)
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
  else
    img_gray = img;

  seeta::ImageData image_data;
  image_data.data = img_gray.data;
  image_data.width = img_gray.cols;
  image_data.height = img_gray.rows;
  image_data.num_channels = 1;

  // Detect faces
  std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
  int32_t face_num = static_cast<int32_t>(faces.size());

  if (face_num == 0)
  {
    return 0;
  }

  // Detect 5 facial landmarks
  seeta::FacialLandmark points[5];
  point_detector.PointDetectLandmarks(image_data, faces[0], points);

  // Visualize the results
  cv::rectangle(img_color, cv::Point(faces[0].bbox.x, faces[0].bbox.y), cv::Point(faces[0].bbox.x + faces[0].bbox.width - 1, faces[0].bbox.y + faces[0].bbox.height - 1), CV_RGB(255, 0, 0));
  for (int i = 0; i<pts_num; i++)
  {
      cv::circle(img_color, cv::Point(points[i].x, points[i].y), 2, CV_RGB(0, 255, 0), cv::FILLED);
  }
  cv::imwrite("result.jpg", img_color);

  // Release memory
  return 0;
}
