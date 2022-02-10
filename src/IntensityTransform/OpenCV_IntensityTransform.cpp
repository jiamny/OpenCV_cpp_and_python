#include <opencv2/opencv.hpp>
#include <opencv2/intensity_transform.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::intensity_transform;

// ����Աȶ�
double rmsContrast(Mat srcImg)
{
	Mat dstImg, dstImg_mean, dstImg_std;
	// �ҶȻ�
	cvtColor(srcImg, dstImg, COLOR_BGR2GRAY);
	// ����ͼ���ֵ�ͷ���
	meanStdDev(dstImg, dstImg_mean, dstImg_std);
	// ���ͼ��Աȶ�
	double contrast = dstImg_std.at<double>(0, 0);
	return contrast;
}

// ����ͼ��
double saveImg(Mat srcImg, String saveType)
{
	String filename = "indicator";
	Mat saveImg = srcImg.clone();
	double contrast = rmsContrast(saveImg);

	putText(saveImg, format("contrast %.3f", contrast), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);
	imwrite("./src/IntensityTransform/" + filename + "_" + saveType + "_result.jpg", saveImg);
	return 0;
}

// ���������ռ������Ⱦ�û�����
namespace
{
	// global variables
	Mat g_image;

	// gamma�任����
	int g_gamma = 40;
	const int g_gammaMax = 500;
	Mat g_imgGamma;
	const std::string g_gammaWinName = "Gamma Correction";

	// �Աȶ�����
	Mat g_contrastStretch;
	int g_r1 = 70;
	int g_s1 = 15;
	int g_r2 = 120;
	int g_s2 = 240;
	const std::string g_contrastWinName = "Contrast Stretching";

	// ����gamma�任������
	static void onTrackbarGamma(int, void*)
	{
		float gamma = g_gamma / 100.0f;
		gammaCorrection(g_image, g_imgGamma, gamma);
		imshow(g_gammaWinName, g_imgGamma);
		cout << g_gammaWinName << ": " << rmsContrast(g_imgGamma) << endl;
		saveImg(g_imgGamma, "g_imgGamma");
	}

	// ���������任������
	static void onTrackbarContrastR1(int, void*)
	{
		contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
		imshow("Contrast Stretching", g_contrastStretch);
		cout << g_contrastWinName << ": " << rmsContrast(g_contrastStretch) << endl;
		saveImg(g_contrastStretch, "g_contrastStretch");
	}

	static void onTrackbarContrastS1(int, void*)
	{
		contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
		imshow("Contrast Stretching", g_contrastStretch);
		cout << g_contrastWinName << ": " << rmsContrast(g_contrastStretch) << endl;
		saveImg(g_contrastStretch, "g_contrastStretch");
	}

	static void onTrackbarContrastR2(int, void*)
	{
		contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
		imshow("Contrast Stretching", g_contrastStretch);
		cout << g_contrastWinName << ": " << rmsContrast(g_contrastStretch) << endl;
		saveImg(g_contrastStretch, "g_contrastStretch");
	}

	static void onTrackbarContrastS2(int, void*)
	{
		contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
		imshow("Contrast Stretching", g_contrastStretch);
		cout << g_contrastWinName << ": " << rmsContrast(g_contrastStretch) << endl;
		saveImg(g_contrastStretch, "g_contrastStretch");
	}
}

int main()
{
	// ͼ��·��
	const std::string inputFilename = "./src/IntensityTransform/image/car.png";

	// Read input image
	// ��ͼ
	g_image = imread(inputFilename);

	if (g_image.empty())
	{
		printf("image is empty");
		return 0;
	}

	// Create trackbars
	// ����������
	namedWindow(g_gammaWinName);
	// ����gamma�任ɸѡ����
	createTrackbar("Gamma value", g_gammaWinName, &g_gamma, g_gammaMax, onTrackbarGamma);
	setTrackbarPos("Gamma value", g_gammaWinName, g_gamma);

	// �Աȶ����� Contrast Stretching
	namedWindow(g_contrastWinName);
	createTrackbar("Contrast R1", g_contrastWinName, &g_r1, 256, onTrackbarContrastR1);
	setTrackbarPos("Contrast R1", g_contrastWinName, g_r1);
	createTrackbar("Contrast S1", g_contrastWinName, &g_s1, 256, onTrackbarContrastS1);
	cv::setTrackbarPos("Contrast S1", g_contrastWinName, g_s1);
	createTrackbar("Contrast R2", g_contrastWinName, &g_r2, 256, onTrackbarContrastR2);
	cv::setTrackbarPos("Contrast R2", g_contrastWinName, g_r2);
	createTrackbar("Contrast S2", g_contrastWinName, &g_s2, 256, onTrackbarContrastS2);
	cv::setTrackbarPos("Contrast S2", g_contrastWinName, g_s2);

	// Apply intensity transformations
	// Ӧ��ǿ��ת��
	Mat imgAutoscaled, imgLog;
	// autoscaling
	autoscaling(g_image, imgAutoscaled);
	// gamma�任
	gammaCorrection(g_image, g_imgGamma, g_gamma / 100.0f);
	// �����任
	logTransform(g_image, imgLog);
	// �Աȶ�����
	contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);

	// Display intensity transformation results
	// չʾ���
	imshow("Original Image", g_image);
	cout << "Original Image: " << rmsContrast(g_image) << endl;
	saveImg(g_image, "g_image");

	imshow("Autoscale", imgAutoscaled);
	cout << "Autoscale: " << rmsContrast(imgAutoscaled) << endl;
	saveImg(imgAutoscaled, "imgAutoscaled");

	imshow(g_gammaWinName, g_imgGamma);
	cout << g_gammaWinName << ": " << rmsContrast(g_imgGamma) << endl;

	imshow("Log Transformation", imgLog);
	cout << "Log Transformation: " << rmsContrast(imgLog) << endl;
	saveImg(imgLog, "imgLog");

	imshow(g_contrastWinName, g_contrastStretch);
	cout << g_contrastWinName << ": " << rmsContrast(g_contrastStretch) << endl;

	waitKey(0);
	return 0;
}
