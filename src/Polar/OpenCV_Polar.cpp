#include <opencv2/opencv.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

// ----- ȫ�ֲ���
// PAIֵ
double PI = M_PI;
// ��������ͼ��̶��ߴ磨��Ҫ��
double HEIGHT = 300;
double WIDTH = 300;
// ����ͼ��Բ�İ뾶��һ���ǿ��һ��
int CIRCLE_RADIUS = int(HEIGHT / 2);
// Բ������
cv::Point CIRCLE_CENTER = cv::Point(int(WIDTH / 2), int(HEIGHT / 2));
// ������ת����ͼ��ĸߣ����Լ�����
int LINE_HEIGHT = int(CIRCLE_RADIUS / 1.5);
// ������ת����ͼ��Ŀ�һ����ԭ��Բ�ε��ܳ�
int LINE_WIDTH = int(2 * CIRCLE_RADIUS * PI);

// C++ OpenCV Mat������ֵ��
typedef Point3_<uint8_t> Pixel;

// ----- ��Բ����Ϊ����
cv::Mat create_line_image(cv::Mat img)
{
	cv::Mat line_image = cv::Mat::zeros(Size(LINE_WIDTH, LINE_HEIGHT), CV_8UC3);
	// �Ƕ�
	double theta;
	// �뾶
	double rho;

	// ����Բ�ļ����긳ֵ
	for (int row = 0; row < line_image.rows; row++)
	{
		for (int col = 0; col < line_image.cols; col++)
		{
			// ����-0.2�������Ż�������������е���
			theta = PI * 2 / LINE_WIDTH * (col + 1) - 0.2;
			rho = CIRCLE_RADIUS - row - 1;

			// ----- �����任
			//int x = int(CIRCLE_CENTER.x + rho * std::cos(theta) + 0);
			//int y = int(CIRCLE_CENTER.y - rho * std::sin(theta) + 0);

			// ----- ������ʼλ�ñ任
			//// 1 ȷ��������
			//double x0 = rho * std::cos(theta) + 0;
			//double y0 = rho * std::sin(theta) + 0;

			//// 2 ȷ����ת�Ƕ�
			//double angle = PI * 2 * (-120.0) / 360;

			//// 3 ȷ��ֱ������
			//double x1 = x0 * std::cos(angle) - y0 * std::sin(angle) + 0;
			//double y1 = x0 * std::sin(angle) + y0 * std::cos(angle) + 0;

			//// 4 �л�ΪOpenCVͼ������
			//int x = int(CIRCLE_CENTER.x + x1);
			//int y = int(CIRCLE_CENTER.y - y1);

			// ----- ������ʼλ��˳ʱ��任
			// 1 ȷ��������
			double x0 = rho * std::sin(theta) + 0;
			double y0 = rho * std::cos(theta) + 0;

			// 2 ȷ����ת�Ƕ�
			double angle = PI * 2 * (-150.0) / 360;

			// 3 ȷ��ֱ������
			double x1 = x0 * std::cos(angle) - y0 * std::sin(angle) + 0;
			double y1 = x0 * std::sin(angle) + y0 * std::cos(angle) + 0;

			// 4 �л�Ϊopencvͼ������
			int x = int(CIRCLE_CENTER.x + x1);
			int y = int(CIRCLE_CENTER.y - y1);

			// Obtain pixel at(y,x)ֱ�ӷ�����������(Ч�ʲ��ߣ������޸ģ�
			Pixel pixel = img.at<Pixel>(y, x);
			// ��ֵ
			line_image.at<Pixel>(row, col) = pixel;
		}
	}
	// �����ı����ͼ������ת������
	// cv::rotate(line_image, line_image, cv::ROTATE_90_CLOCKWISE);
	return line_image;
}

// ----- ������
int main()
{
	// ����ͼ��·��
	String imgpath = "./src/Polar/image/clock.jpg";
	// ��ȡͼ��
	cv::Mat img = cv::imread(imgpath);
	if (img.empty())
	{
		printf("please check image path");
		return -1;
	}
	// ͼ������Ϊ�̶���С
	cv::resize(img, img, Size(WIDTH, HEIGHT));
	printf("shape is: %d,%d", img.rows, img.cols);
	// չʾԭͼ
	cv::imshow("src", img);
	cv::Mat output = create_line_image(img);
	// չʾ���
	cv::imshow("dst", output);
	cv::waitKey();
	cv::destroyAllWindows();
	system("pause");
	return 0;
}
