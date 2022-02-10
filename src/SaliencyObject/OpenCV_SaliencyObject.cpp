#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace saliency;

int main()
{
	// �����Լ���㷨
	// ��ѡ��SPECTRAL_RESIDUAL��FINE_GRAINED��BING��BinWangApr2014
	String saliency_algorithm = "BING";
	// �����Ƶ����ͼ��
	String video_name = "./src/SaliencyObject/video/vtest.avi";
	// String video_name = "video/dog.jpg";
	// ��ʼ֡
	int start_frame = 0;
	// ģ��·��
	String training_path = "ObjectnessTrainedModel";

	// ����㷨������Ƶ��Ϊ�գ�ֹͣ���
	if (saliency_algorithm.empty() || video_name.empty())
	{
		cout << "Please set saliency_algorithm and video_name";
		return -1;
	}

	// open the capture
	VideoCapture cap;
	// ����Ƶ
	cap.open(video_name);
	// ������Ƶ��ʼ֡
	cap.set(CAP_PROP_POS_FRAMES, start_frame);

	// ����ͼ��
	Mat frame;

	// instantiates the specific Saliency
	// ʵ����saliencyAlgorithm�ṹ
	Ptr<Saliency> saliencyAlgorithm;

	// ��ֵ�������
	Mat binaryMap;
	// ���ͼ��
	Mat image;

	// ��ͼ
	cap >> frame;
	if (frame.empty())
	{
		return 0;
	}

	frame.copyTo(image);

	// ��������ķ���ȷ���������
	// StaticSaliencySpectralResidual
	if (saliency_algorithm.find("SPECTRAL_RESIDUAL") == 0)
	{
		// ���������ɫ�����ʾ��������
		Mat saliencyMap;
		saliencyAlgorithm = StaticSaliencySpectralResidual::create();
		// ����������
		double start = static_cast<double>(getTickCount());
		bool success = saliencyAlgorithm->computeSaliency(image, saliencyMap);
		double duration = ((double)getTickCount() - start) / getTickFrequency();
		cout << "computeSaliency cost time is: " << duration * 1000 << "ms" << endl;

		if (success)
		{
			StaticSaliencySpectralResidual spec;
			// ��ֵ��ͼ��
			double start = static_cast<double>(getTickCount());
			spec.computeBinaryMap(saliencyMap, binaryMap);
			double duration = ((double)getTickCount() - start) / getTickFrequency();
			cout << "computeBinaryMap cost time is: " << duration * 1000 << "ms" << endl;

			imshow("Original Image", image);
			imshow("Saliency Map", saliencyMap);
			imshow("Binary Map", binaryMap);

			// ת����ʽ���ܱ���ͼƬ
			saliencyMap.convertTo(saliencyMap, CV_8UC3, 256);
			imwrite("./src/SaliencyObject/Results/SPECTRAL_RESIDUAL_saliencyMap.jpg", saliencyMap);
			imwrite("./src/SaliencyObject/Results/SPECTRAL_RESIDUAL_binaryMap.jpg", binaryMap);
			waitKey(0);
		}
	}

	// StaticSaliencyFineGrained
	else if (saliency_algorithm.find("FINE_GRAINED") == 0)
	{
		Mat saliencyMap;
		saliencyAlgorithm = StaticSaliencyFineGrained::create();
		// ����������
		double start = static_cast<double>(getTickCount());
		bool success = saliencyAlgorithm->computeSaliency(image, saliencyMap);
		double duration = ((double)getTickCount() - start) / getTickFrequency();
		cout << "computeSaliency cost time is: " << duration * 1000 << "ms" << endl;

		if (success)
		{
			StaticSaliencyFineGrained spec;
			// ��ֵ��ͼ��
			double start = static_cast<double>(getTickCount());
			spec.computeBinaryMap(saliencyMap, binaryMap);
			double duration = ((double)getTickCount() - start) / getTickFrequency();
			cout << "computeBinaryMap cost time is: " << duration * 1000 << "ms" << endl;

			imshow("Saliency Map", saliencyMap);
			imshow("Original Image", image);
			imshow("Binary Map", binaryMap);

			// ת����ʽ���ܱ���ͼƬ
			saliencyMap.convertTo(saliencyMap, CV_8UC3, 256);
			imwrite("./src/SaliencyObject/Results/FINE_GRAINED_saliencyMap.jpg", saliencyMap);
			imwrite("./src/SaliencyObject/Results/FINE_GRAINED_binaryMap.jpg", binaryMap);
			waitKey(0);
		}
	}

	// ObjectnessBING
	else if (saliency_algorithm.find("BING") == 0)
	{
		// �ж�ģ���Ƿ����
		if (training_path.empty())
		{
			cout << "Path of trained files missing! " << endl;
			return -1;
		}

		else
		{
			saliencyAlgorithm = ObjectnessBING::create();
			vector<Vec4i> saliencyMap;
			// ��ȡģ���ļ�����
			saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setTrainingPath(training_path);
			// ���㷨�����������Results�ļ�����
			saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setBBResDir("Results");
			// ���÷Ǽ���ֵ���ƣ�ֵԽ���⵽��Ŀ��Խ�٣�����ٶ�Խ��
			saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setNSS(50);

			// ����������
			double start = static_cast<double>(getTickCount());
			// ����������ɫ�ռ���м�⣬����ֻ���һ���ռ䣬��training_path�������ռ�ģ��ɾ������
			// ��ֻ����ObjNessB2W8MAXBGRǰ׺���ļ����㷨��ʱֻ��ԭ����һ��
			bool success = saliencyAlgorithm->computeSaliency(image, saliencyMap);
			double duration = ((double)getTickCount() - start) / getTickFrequency();
			cout << "computeSaliency cost time is: " << duration * 1000 << "ms" << endl;

			if (success)
			{
				// saliencyMap��ȡ��⵽��Ŀ�����
				int ndet = int(saliencyMap.size());
				std::cout << "Objectness done " << ndet << std::endl;
				// The result are sorted by objectness. We only use the first maxd boxes here.
				// Ŀ�갴�����ԴӴ�С���У�maxdΪ��ʾǰ5��Ŀ�꣬step������ɫ��jitter���þ��ο�΢��
				int maxd = 5, step = 255 / maxd, jitter = 9;
				Mat draw = image.clone();
				for (int i = 0; i < std::min(maxd, ndet); i++)
				{
					// ��þ��ο������
					Vec4i bb = saliencyMap[i];
					// �趨��ɫ
					Scalar col = Scalar(((i*step) % 255), 50, 255 - ((i*step) % 255));
					// ���ο�΢��
					Point off(theRNG().uniform(-jitter, jitter), theRNG().uniform(-jitter, jitter));
					// ������
					rectangle(draw, Point(bb[0] + off.x, bb[1] + off.y), Point(bb[2] + off.x, bb[3] + off.y), col, 2);
					// mini temperature scale
					// ��ɫ��ע
					rectangle(draw, Rect(20, 20 + i * 10, 10, 10), col, -1);
				}
				imshow("BING", draw);

				// ����ͼƬ
				imwrite("./src/SaliencyObject/Results/BING_draw.jpg", draw);
				waitKey();
			}
			else
			{
				std::cout << "No saliency found for " << video_name << std::endl;
			}
		}
	}

	// BinWangApr2014
	else if (saliency_algorithm.find("BinWangApr2014") == 0)
	{
		saliencyAlgorithm = MotionSaliencyBinWangApr2014::create();
		// �������ݽṹ��С
		saliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->setImagesize(image.cols, image.rows);
		// ��ʼ��
		saliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->init();

		bool paused = false;
		int i = 0;
		for (;; )
		{
			if (!paused)
			{
				cap >> frame;
				if (frame.empty())
				{
					return 0;
				}
				Mat srcImg = frame.clone();
				cvtColor(frame, frame, COLOR_BGR2GRAY);

				Mat saliencyMap;
				// ����
				double start = static_cast<double>(getTickCount());
				saliencyAlgorithm->computeSaliency(frame, saliencyMap);
				double duration = ((double)getTickCount() - start) / getTickFrequency();
				cout << "computeSaliency cost time is: " << duration * 1000 << "ms" << endl;

				imshow("image", frame);
				// ��ʾ
				imshow("saliencyMap", saliencyMap * 255);

				i++;
				if (i == 100)
				{
					imwrite("./src/SaliencyObject/Results/origin.jpg", srcImg);
					// ת����ʽ���ܱ���ͼƬ
					saliencyMap.convertTo(saliencyMap, CV_8UC3, 256);
					imwrite("./src/SaliencyObject/Results/BinWangApr2014_saliencyMap.jpg", saliencyMap);
				}
			}

			char c = (char)waitKey(2);
			if (c == 'q')
				break;
			if (c == 'p')
				paused = !paused;
		}
	}

	destroyAllWindows();
	return 0;
}
