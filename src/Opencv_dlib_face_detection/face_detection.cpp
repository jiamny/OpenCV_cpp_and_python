#include "face_detection.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

string faceCascadePath = "./models/haarcascade_frontalface_default.xml";

string caffeConfigFile = "./models/deploy.prototxt";
string caffeWeightFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
string tensorflowConfigFile = "./models/opencv_face_detector.pbtxt";
string tensorflowWeightFile = "./models/opencv_face_detector_uint8.pb";

int main(void) {
	Mat frame = imread("./src/Opencv_dlib_face_detection/1.jpg");
	Mat resultImg_Harr = detectFaceHaar(frame, faceCascadePath);
	Mat resultImg_OpenCVCaffe = detectFaceOpenCVDNN(frame, caffeConfigFile, caffeWeightFile);
	Mat resultImg_OpenCVTf = detectFaceOpenCVDNN(frame, tensorflowConfigFile, tensorflowWeightFile);

	imshow("origin img", frame);
	waitKey(0);
	imshow("Harr", resultImg_Harr);
	waitKey(0);
	imshow("OpenCVCaffe", resultImg_OpenCVCaffe);
	waitKey(0);
	imshow("OpenCVTf", resultImg_OpenCVTf);
	waitKey(0);

	return 0;
}



