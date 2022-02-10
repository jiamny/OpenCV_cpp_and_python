// ��Ƶ���ַŴ�����

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

int main()
{
	string input_path = string("./src/ImageSuperres/video/chaplin.mp4");
	string output_path = string("./src/ImageSuperres/video/out_chaplin.mp4");
	// ѡ��ģ�� edsr, espcn, fsrcnn or lapsrn
	string algorithm = string("lapsrn");
	// �Ŵ������2��3��4��8������ģ�ͽṹѡ��
	int scale = 2;
	// ģ��·��
	string path = string("./models/ImageSuperres/LapSRN_x2.pb");

	// ����Ƶ
	VideoCapture input_video(input_path);
	// ����ͼ�����ߴ�
	int ex = static_cast<int>(input_video.get(CAP_PROP_FOURCC));
	// ��������Ƶͼ��ߴ�
	Size S = Size((int)input_video.get(CAP_PROP_FRAME_WIDTH) * scale,
		(int)input_video.get(CAP_PROP_FRAME_HEIGHT) * scale);

	VideoWriter output_video;
	output_video.open(output_path, ex, input_video.get(CAP_PROP_FPS), S, true);

	// �����Ƶû�д�
	if (!input_video.isOpened())
	{
		std::cerr << "Could not open the video." << std::endl;
		return -1;
	}

	// ��ȡ���ַŴ�ģ��
	DnnSuperResImpl sr;
	sr.readModel(path);
	sr.setModel(algorithm, scale);

	for (;;)
	{
		Mat frame, output_frame;
		input_video >> frame;

		if (frame.empty())
			break;

		// �ϲ���ͼ��
		sr.upsample(frame, output_frame);
		output_video << output_frame;

		namedWindow("Upsampled video", WINDOW_AUTOSIZE);
		imshow("Upsampled video", output_frame);

		namedWindow("Original video", WINDOW_AUTOSIZE);
		imshow("Original video", frame);

		char c = (char)waitKey(1);
		// esc�˳�
		if (c == 27)
		{
			break;
		}
	}

	input_video.release();
	output_video.release();

	return 0;
}
