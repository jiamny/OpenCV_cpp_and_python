#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>

#include <iostream>

using namespace cv;
using namespace cv::img_hash;
using namespace std;

template <typename T>
inline void test_one(const std::string &title, const Mat &a, const Mat &b)
{
	cout << "=== " << title << " ===" << endl;
	TickMeter tick;
	Mat hashA, hashB;
	// ģ�巽���ظ�����
	Ptr<ImgHashBase> func;
	func = T::create();

	tick.reset();
	tick.start();
	// ����ͼa�Ĺ�ϣֵ
	func->compute(a, hashA);
	tick.stop();
	cout << "compute1: " << tick.getTimeMilli() << " ms" << endl;

	tick.reset();
	tick.start();
	// ����ͼb�Ĺ�ϣֵ
	func->compute(b, hashB);
	tick.stop();
	cout << "compute2: " << tick.getTimeMilli() << " ms" << endl;

	// �Ƚ�����ͼ���ϣֵ�ľ���
	cout << "compare: " << func->compare(hashA, hashB) << endl << endl;
}

int main()
{
	// ������ͼ��������ƶȱȽ�
	Mat input = imread("./src/ImgHash/image/img1.jpg");
	Mat target = imread("./src/ImgHash/image/img4.jpg");

	// ͨ����ͬ�����Ƚ�ͼ��������
	test_one<AverageHash>("AverageHash", input, target);
	test_one<PHash>("PHash", input, target);
	test_one<MarrHildrethHash>("MarrHildrethHash", input, target);
	test_one<RadialVarianceHash>("RadialVarianceHash", input, target);
	test_one<BlockMeanHash>("BlockMeanHash", input, target);
	test_one<ColorMomentHash>("ColorMomentHash", input, target);

	system("pause");
	return 0;
}
