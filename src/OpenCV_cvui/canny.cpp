
#include <opencv2/opencv.hpp>
#define CVUI_IMPLEMENTATION
#include "cvui.h"

//cvui��������
#define WINDOW_NAME	"CVUI Canny Edge"

int main()
{
	//��ͼ��
	cv::Mat lena = cv::imread("./src/OpenCV_cvui/image/lena.jpg");
	//����ͼ��
	cv::Mat frame = lena.clone();
	//canny��ֵ
	int low_threshold = 50, high_threshold = 150;
	//�Ƿ�ʹ�ñ�Ե���
	bool use_canny = false;

	// Init a OpenCV window and tell cvui to use it.
	// If cv::namedWindow() is not used, mouse events will
	// not be captured by cvui.
	//����cvui����
	cv::namedWindow(WINDOW_NAME);
	//��ʼ������
	cvui::init(WINDOW_NAME);

	while (true)
	{
		// Should we apply Canny edge?
		//�Ƿ�ʹ�ñ�Ե���
		if (use_canny) {
			// Yes, we should apply it.
			cv::cvtColor(lena, frame, cv::COLOR_BGR2GRAY);
			cv::Canny(frame, frame, low_threshold, high_threshold, 3);
			cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
		} else {
			// No, so just copy the original image to the displaying frame.
			//ֱ����ʾͼ��
			lena.copyTo(frame);
		}

		// Render the settings window to house the checkbox
		// and the trackbars below.
		//debug�¿�����bug
		//��Ҫ��������cvui.h��void window�������⣬����취aOverlay = theBlock.where.clone();
		//��frame(10,50)������һ������180��180����ΪSettings����
		cvui::window(frame, 10, 50, 180, 180, "Settings");
		
		// Checkbox to enable/disable the use of Canny edge
		//��frame(15,80)����Ӹ�ѡ�򣬸�ѡ���ı���"Use Canny Edge"����������use_canny
		cvui::checkbox(frame, 15, 80, "Use Canny Edge", &use_canny);

		// Two trackbars to control the low and high threshold values
		// for the Canny edge algorithm
		//������������ͷָ���ֵ
		//��frame(15,110)����ӻ���������������165������ֵlow_threshold��ֵ�仯��Χ5��150
		cvui::trackbar(frame, 15, 110, 165, &low_threshold, 5, 150);
		//������������߷ָ���ֵ
		cvui::trackbar(frame, 15, 180, 165, &high_threshold, 80, 300);

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		//����ui����
		cvui::update();

		// Show everything on the screen
		//�����еĶ�����ʾ����
		cv::imshow(WINDOW_NAME, frame);

		// Check if ESC was pressed
		//ESC�˳�
		if (cv::waitKey(30) == 27) {
			break;
		}
	}

	return 0;
}
