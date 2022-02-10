#include "face_detection.h"

//检测图像宽高
const size_t inWidth = 300;
const size_t inHeight = 300;
//缩放比例
const double inScaleFactor = 1.0;
//阈值
const double confidenceThreshold = 0.7;
//均值
const cv::Scalar meanVal(104.0, 177.0, 123.0);

/*
 * @brief 人脸检测Opencv ssd
 *
 * @param frame 原图
 * @param configFile 模型结构定义文件
 * @param weightFile 模型文件
 * @return Mat
 */
Mat detectFaceOpenCVDNN(Mat frame, string configFile, string weightFile)
{
	Mat frameOpenCVDNN = frame.clone();
	Net net;
	Mat inputBlob;
	int frameHeight = frameOpenCVDNN.rows;
	int frameWidth = frameOpenCVDNN.cols;
	//获取文件后缀
	string suffixStr = configFile.substr(configFile.find_last_of('.') + 1);
	//判断是caffe模型还是tensorflow模型
	if (suffixStr == "prototxt")
	{
		net = dnn::readNetFromCaffe(configFile, weightFile);
		inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
	}
	else
	{
		//bug
		//net = dnn::readNetFromTensorflow(configFile, weightFile);
		net = dnn::readNet(configFile, weightFile);
		inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
	}

	//读图检测
	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	for (int i = 0; i < detectionMat.rows; i++)
	{
		//分类精度
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidenceThreshold)
		{
			//左上角坐标
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
			//右下角坐标
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
			//画框
			cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
		}
	}
	return frameOpenCVDNN;
}




