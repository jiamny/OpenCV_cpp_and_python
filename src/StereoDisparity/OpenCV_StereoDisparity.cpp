#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>

int main()
{
	cv::Mat imgL, imgR;

	// ��ͼ
	imgL = cv::imread("./src/StereoDisparity/images/im0.png", 0);
	cv::resize(imgL, imgL, cv::Size(600, 600), cv::INTER_LINEAR);
	imgR = cv::imread("./src/StereoDisparity/images/im1.png", 0);
	cv::resize(imgR, imgR, cv::Size(600, 600));

	int minDisparity = 0;
	int numDisparities = 64;
	int blockSize = 8;
	int disp12MaxDiff = 1;
	int uniquenessRatio = 10;
	int speckleWindowSize = 10;
	int speckleRange = 8;

	// ����StereoSGBM����
	cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize,
			disp12MaxDiff, uniquenessRatio, speckleWindowSize, speckleRange);

	cv::Mat disp;
	// �����Ӳ�
	stereo->compute(imgL, imgR, disp);

	// �����һ��
	cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	cv::imshow("Left image", imgL);
	cv::imshow("Right image", imgR);
	cv::imshow("disparity", disp);
	cv::waitKey(0);

	return 0;
}
