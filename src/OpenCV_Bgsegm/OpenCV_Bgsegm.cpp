#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>
#include <iostream>

using namespace cv;
using namespace cv::bgsegm;

const String algos[7] = { "GMG", "CNT", "KNN", "MOG", "MOG2", "GSOC", "LSBP" };

// ������ͬ�ı����ָ�ʶ����
static Ptr<BackgroundSubtractor> createBGSubtractorByName(const String& algoName)
{
	Ptr<BackgroundSubtractor> algo;
	if (algoName == String("GMG"))
		algo = createBackgroundSubtractorGMG(20, 0.7);
	else if (algoName == String("CNT"))
		algo = createBackgroundSubtractorCNT();
	else if (algoName == String("KNN"))
		algo = createBackgroundSubtractorKNN();
	else if (algoName == String("MOG"))
		algo = createBackgroundSubtractorMOG();
	else if (algoName == String("MOG2"))
		algo = createBackgroundSubtractorMOG2();
	else if (algoName == String("GSOC"))
		algo = createBackgroundSubtractorGSOC();
	else if (algoName == String("LSBP"))
		algo = createBackgroundSubtractorLSBP();

	return algo;
}

int main() {
	// ��Ƶ·��
	String videoPath = "./src/OpenCV_Bgsegm/video/vtest.avi";

	// �����ָ�ʶ�������
	int algo_index = 0;
	// ���������ָ�ʶ����
	Ptr<BackgroundSubtractor> bgfs = createBGSubtractorByName(algos[algo_index]);

	// ����Ƶ
	VideoCapture cap;
	cap.open(videoPath);

	// �����Ƶû�д�
	if (!cap.isOpened()) {
		std::cerr << "Cannot read video. Try moving video file to sample directory." << std::endl;
		return -1;
	}

	// ����ͼ��
	Mat frame;
	// �˶�ǰ��
	Mat fgmask;
	// �����ʾ��ͼ��
	Mat segm;

	// �ӳٵȴ�ʱ��
	int delay = 30;
	// ������л���CPU�ĺ�����
	int nthreads = getNumberOfCPUs();
	// �����߳���
	setNumThreads(nthreads);

	// �Ƿ���ʾ�˶�ǰ��
	bool show_fgmask = false;

	// ƽ��ִ��ʱ��
	float average_Time = 0.0;
	// ��ǰ֡��
	int frame_num = 0;
	// ��ִ��ʱ��
	float sum_Time = 0.0;

	for (;;) {
		// ��ȡ֡
		cap >> frame;

		// ���ͼƬΪ��
		if(frame.empty()) {
			// CAP_PROP_POS_FRAMES��ʾ��ǰ֡
			// ���仰��ʾ����ǰ֡�趨Ϊ��0֡
			cap.set(CAP_PROP_POS_FRAMES, 0);
			cap >> frame;
		}

		double time0 = static_cast<double>(getTickCount());

		// ������ģ
		bgfs->apply(frame, fgmask);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		// ��ִ��ʱ��
		sum_Time += time0;
		// ƽ��ÿִ֡��ʱ��
		average_Time = sum_Time / (frame_num + 1);

		if (show_fgmask) {
			segm = fgmask;
		} else {
			// ����segm = alpha * frame + beta�ı�ͼƬ
			// �����ֱ�Ϊ�����ͼ�����ͼ���ʽ��alphaֵ��betaֵ
			frame.convertTo(segm, CV_8U, 0.5);
			// ͼ�����
			// �����ֱ�Ϊ������ͼ��/��ɫ1������ͼ��/��ɫ2�����ͼ����Ĥ
			// ��Ĥ��ʾ���ӷ�Χ
			add(frame, Scalar(100, 100, 0), segm, fgmask);
		}

		// ��ʾ��ǰ����
		cv::putText(segm, algos[algo_index], Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);
		// ��ʾ��ǰ�߳���
		cv::putText(segm, format("%d threads", nthreads), Point(10, 60), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);
		// ��ʾ��ǰÿִ֡��ʱ��
		cv::putText(segm, format("averageTime %f s", average_Time), Point(10, 90), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);

		cv::imshow("FG Segmentation", segm);

		int c = waitKey(delay);

		// �޸ĵȴ�ʱ��
		if (c == ' ') {
			delay = delay == 30 ? 1 : 30;
		}

		// ��C�����ָ�ʶ����
		if (c == 'c' || c == 'C') {
			algo_index++;
			if (algo_index > 6)
				algo_index = 0;

			bgfs = createBGSubtractorByName(algos[algo_index]);
		}

		// �����߳���
		if (c == 'n' || c == 'N') {
			nthreads++;
			if (nthreads > 8)
				nthreads = 1;

			setNumThreads(nthreads);
		}

		// �Ƿ���ʾ����
		if (c == 'm' || c == 'M') {
			show_fgmask = !show_fgmask;
		}

		// �˳�
		if (c == 'q' || c == 'Q' || c == 27) {
			break;
		}

		// ��ǰ֡������
		frame_num++;
		if (100 == frame_num) {
			String strSave = "./src/OpenCV_Bgsegm/out_" + algos[algo_index] + ".jpg";
			imwrite(strSave, segm);
		}
	}

	return 0;
}
