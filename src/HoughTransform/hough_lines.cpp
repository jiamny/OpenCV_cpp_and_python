
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

// variables to store images
Mat dst, cimg, gray, img, edges;

int initThresh;
const int maxThresh = 1000;
double th1, th2;

// create a vector to store points of line
vector<Vec4i> lines;

void onTrackbarChange(int, void*)
{
	//����Ŀ��ͼ��
	cimg = img.clone();
	//���ͼ��
	dst = img.clone();

	th1 = initThresh;
	th2 = th1 * 0.4;
	//canny��Ե���
	Canny(img, edges, th1, th2);

	// apply hough line transform �������任
	HoughLinesP(edges, lines, 2, CV_PI / 180, 50, 10, 100);

	// draw lines on the detected points ����
	for (size_t i = 0; i < lines.size(); i++)
	{
		//��ȡ���������
		Vec4i l = lines[i];
		line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
	}

	// show the resultant image
	imshow("Result Image", dst);
	imshow("Edges", edges);
}

int main()
{
	// Read image (color mode) ��ͼ
	img = imread("./src/HoughTransform/image/lanes.jpg", 1);
	dst = img.clone();

	if (img.empty())
	{
		cout << "Error in reading image" << endl;
		return -1;
	}

	// Convert to gray-scale ת��Ϊ�Ҷ�ͼ��
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// Detect edges using Canny Edge Detector
	// Canny(gray, dst, 50, 200, 3);

	// Make a copy of original image
	// cimg = img.clone();

	// Will hold the results of the detection
	namedWindow("Edges", 1);
	namedWindow("Result Image", 1);

	// Declare thresh to vary the max_radius of circles to be detected in hough transform
	// �������任��ֵ
	initThresh = 500;

	// Create trackbar to change threshold values
	//������
	createTrackbar("threshold", "Result Image", &initThresh, maxThresh, onTrackbarChange);
	onTrackbarChange(initThresh, 0);

	while (true)
	{
		int key;
		key = waitKey(1);
		if ((char)key == 27)
		{
			break;
		}
	}
	destroyAllWindows();
	return 0;
}
