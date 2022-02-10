#include "pch.h"
#include <opencv2/opencv.hpp>
#define CVUI_IMPLEMENTATION
#include "cvui.h"

//cvui��������
#define WINDOW_NAME "CVUI Hello World!"

int main()
{
	cv::Mat frame = cv::Mat(200, 500, CV_8UC3);
	int count = 0;

	// Init a OpenCV window and tell cvui to use it.
	//����cvui����
	cv::namedWindow(WINDOW_NAME);
	//��ʼ������
	cvui::init(WINDOW_NAME);

	//����Ҫ������ѭ����ÿ�α䶯cvui�������µ�һ��ͼ�񣬿���������仯��
	while (true)
	{
		// Fill the frame with a nice color �������򴰿ڱ���ͼ��
		frame = cv::Scalar(49, 52, 49);

		// Buttons will return true if they were clicked
		//�ڱ���ͼ��(110,80)����Ӱ�ť(��ť�����ϽǶ������꣬���е�cvui���궼�����ϽǶ���)����ť��ʾ����Ϊ��hello,world��
		//����ť�����ʱ���᷵��true
		if (cvui::button(frame, 110, 80, "Hello, world!"))
		{
			// The button was clicked, so let's increment our counter.
			//ͳ�ư�ť���������
			count++;
		}

		// Sometimes you want to show text that is not that simple, e.g. strings + numbers.
		// You can use cvui::printf for that. It accepts a variable number of parameter, pretty
		// much like printf does.
		// Let's show how many times the button has been clicked.
		//��frame(250,90)�����һ���ı����ı��������СΪ0.5,��ɫΪ0xff0000
		//��ʾ������Ϊ"Button click count: %d", count
		cvui::printf(frame, 250, 90, 0.5, 0xff0000, "Button click count: %d", count);

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		//����cvui����
		cvui::update();

		// Show everything on the screen
		//�����еĶ�����ʾ����
		cv::imshow(WINDOW_NAME, frame);
		// Check if ESC key was pressed
		//ESC�˳�ѭ��
		if (cv::waitKey(20) == 27)
		{
			break;
		}
	}

	return 0;
}