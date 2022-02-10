// ͼ�񳬷ַŴ�����
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

int main()
{
	// ͼ��·��
	string img_path = string("./src/ImageSuperres/image/image.png");
	if (img_path.empty())
	{
		printf("image is empty!");
	}
	// ��ѡ������Ŵ����2��4��8��','�ָ��Ŵ����
	string scales_str = string("2,4,8");
	// ��ѡģ������Ŵ���������NCHW_output_2x,NCHW_output_4x��NCHW_output_8x
	// ��Ҫ����ģ�ͺ�����Ŵ������ͬȷ��ȷ��
	string output_names_str = string("NCHW_output_2x,NCHW_output_4x,NCHW_output_8x");
	// ģ��·��
	std::string path = string("./models/ImageSuperres/model/LapSRN_x8.pb");

	// Parse the scaling factors
	// �����Ŵ��������
	std::vector<int> scales;
	char delim = ',';
	{
		std::stringstream ss(scales_str);
		std::string token;
		while (std::getline(ss, token, delim))
		{
			scales.push_back(atoi(token.c_str()));
		}
	}

	// Parse the output node names
	// ����ģ�ͷŴ�����
	std::vector<String> node_names;
	{
		std::stringstream ss(output_names_str);
		std::string token;
		while (std::getline(ss, token, delim))
		{
			node_names.push_back(token);
		}
	}

	// Load the image
	// ����ͼƬ
	Mat img = cv::imread(img_path);
	Mat original_img(img);
	if (img.empty())
	{
		std::cerr << "Couldn't load image: " << img << "\n";
		return -2;
	}

	// Make dnn super resolution instance
	// ����Dnn Superres����
	DnnSuperResImpl sr;
	// ������Ŵ����
	int scale = *max_element(scales.begin(), scales.end());
	std::vector<Mat> outputs;
	// ��ȡģ��
	sr.readModel(path);
	// �趨ģ�����
	sr.setModel("lapsrn", scale);
	// ��������ַŴ�ͼ��
	sr.upsampleMultioutput(img, outputs, scales, node_names);

	for (unsigned int i = 0; i < outputs.size(); i++)
	{
		cv::namedWindow("Upsampled image", WINDOW_AUTOSIZE);
		// ��ͼ����ʾ��ǰ�Ŵ����
		cv::putText(outputs[i], format("Scale %d", scales[i]), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);
		cv::imshow("Upsampled image", outputs[i]);
		cv::imwrite("./src/ImageSuperres/" + to_string(i) + ".jpg", outputs[i]);
		cv::waitKey(-1);
	}

	return 0;
}
