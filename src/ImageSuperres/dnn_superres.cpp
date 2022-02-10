// ͼ�񳬷ַŴ����

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

int main()
{
	string img_path = string("./src/ImageSuperres/image/image.png");
	// ��ѡ���㷨��bilinear, bicubic, edsr, espcn, fsrcnn or lapsrn
	string algorithm = string("lapsrn");
	// �Ŵ������������ֵ2��3��4
	int scale = 4;
	// ģ��·��
	string path = "./models/ImageSuperres/LapSRN_x4.pb";

	// Load the image
	// ����ͼ��
	Mat img = cv::imread(img_path);
	// ��������ͼ��Ϊ��
	if (img.empty())
	{
		std::cerr << "Couldn't load image: " << img << "\n";
		return -2;
	}

	Mat original_img(img);
	// Make dnn super resolution instance
	// ����dnn���ֱ��ʶ���
	DnnSuperResImpl sr;

	// ���ַŴ���ͼ��
	Mat img_new;

	// ˫���Բ�ֵ
	if (algorithm == "bilinear")
	{
		resize(img, img_new, Size(), scale, scale, cv::INTER_LINEAR);
	}
	// ˫���β�ֵ
	else if (algorithm == "bicubic")
	{
		resize(img, img_new, Size(), scale, scale, cv::INTER_CUBIC);
	}
	else if (algorithm == "edsr" || algorithm == "espcn" || algorithm == "fsrcnn" || algorithm == "lapsrn")
	{
		// ��ȡģ��
		sr.readModel(path);
		// �趨�㷨�ͷŴ����
		sr.setModel(algorithm, scale);
		// �Ŵ�ͼ��
		sr.upsample(img, img_new);
	}
	else
	{
		std::cerr << "Algorithm not recognized. \n";
	}

	// ���ʧ��
	if (img_new.empty())
	{
		// �Ŵ�ʧ��
		std::cerr << "Upsampling failed. \n";
		return -3;
	}
	cout << "Upsampling succeeded. \n";

	// Display image
	// չʾͼƬ
	cv::namedWindow("Initial Image", WINDOW_AUTOSIZE);
	// ��ʼ��ͼƬ
	cv::imshow("Initial Image", img_new);
	cv::imwrite("./src/ImageSuperres/saved.jpg", img_new);
	cv::waitKey(0);

	return 0;
}
