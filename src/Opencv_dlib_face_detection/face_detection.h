

#ifndef SRC_OPENCV_DLIB_FACE_DETECTION_FACE_DETECTION_H_
#define SRC_OPENCV_DLIB_FACE_DETECTION_FACE_DETECTION_H_

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

Mat detectFaceHaar(Mat frame, string faceCascadePath);
Mat detectFaceOpenCVDNN(Mat frame, string configFile, string weightFile);



#endif /* SRC_OPENCV_DLIB_FACE_DETECTION_FACE_DETECTION_H_ */
