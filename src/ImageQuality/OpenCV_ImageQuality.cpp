#include <opencv2/opencv.hpp>
#include <opencv2/quality.hpp>

using namespace std;
using namespace cv;

// ��������ֵ
double calMEAN(Scalar result)
{
	int i = 0;
	double sum = 0;
	// �����ܺ�
	for (auto val : result.val)
	{
		if (0 == val || isinf(val))
		{
			break;
		}
		sum += val;
		i++;
	}
	return sum / i;
}

// ������� MSE
double MSE(Mat img1, Mat img2)
{
	// output quality map
	// �������ͼ
	// �������ͼquality_map���Ǽ��ͼ��ͻ�׼ͼ��������ص��ֵͼ��
	cv::Mat quality_map;
	// compute MSE via static method
	// cv::noArray() if not interested in output quality maps
	// ��̬������һ����λ
	// ����������������ͼ����quality_map�滻ΪnoArray()
	cv::Scalar result_static = quality::QualityMSE::compute(img1, img2, quality_map);

	/* ����һ�ֶ�̬����ķ���
	// alternatively, compute MSE via instance
	cv::Ptr<quality::QualityBase> ptr = quality::QualityMSE::create(img1);
	// compute MSE, compare img1 vs img2
	cv::Scalar result = ptr->compute(img2);
	ptr->getQualityMap(quality_map);
	*/

	return calMEAN(result_static);
}

// ��ֵ����� PSNR
double PSNR(Mat img1, Mat img2)
{
	// �������ͼ
	// �������ͼquality_map���Ǽ��ͼ��ͻ�׼ͼ��������ص��ֵͼ��
	cv::Mat quality_map;
	// ��̬������һ����λ
	// ����������������ͼ����quality_map�滻ΪnoArray()
	// ���ĸ�����ΪPSNR���㹫ʽ�е�MAX����ͼƬ���ܵ��������ֵ��ͨ��Ϊ255
	cv::Scalar result_static = quality::QualityPSNR::compute(img1, img2, quality_map, 255.0);

	/* ����һ�ֶ�̬����ķ���
	cv::Ptr<quality::QualityBase> ptr = quality::QualityPSNR::create(img1, 255.0);
	cv::Scalar result = ptr->compute(img2);
	ptr->getQualityMap(quality_map);*/

	return calMEAN(result_static);
}

// �ݶȷ���������ƫ�� GMSD
double GMSD(Mat img1, Mat img2)
{
	// �������ͼ
	// �������ͼquality_map���Ǽ��ͼ��ͻ�׼ͼ��������ص��ֵͼ��
	cv::Mat quality_map;
	// ��̬������һ����λ
	// ����������������ͼ����quality_map�滻ΪnoArray()
	cv::Scalar result_static = quality::QualityGMSD::compute(img1, img2, quality_map);
	/* ����һ�ֶ�̬����ķ���
	cv::Ptr<quality::QualityBase> ptr = quality::QualityGMSD::create(img1);
	cv::Scalar result = ptr->compute(img2);
	ptr->getQualityMap(quality_map);*/
	return calMEAN(result_static);
}

// �ṹ������ SSIM
double SSIM(Mat img1, Mat img2)
{
	// �������ͼ
	// �������ͼquality_map���Ǽ��ͼ��ͻ�׼ͼ��������ص��ֵͼ��
	cv::Mat quality_map;
	// ��̬������һ����λ
	// ����������������ͼ����quality_map�滻ΪnoArray()
	cv::Scalar result_static = quality::QualitySSIM::compute(img1, img2, quality_map);
	/* ����һ�ֶ�̬����ķ���
	cv::Ptr<quality::QualityBase> ptr = quality::QualitySSIM::create(img1);
	cv::Scalar result = ptr->compute(img2);
	ptr->getQualityMap(quality_map);*/
	return calMEAN(result_static);
}

// ä/�޲ο�ͼ��ռ����������� BRISQUE
double BRISQUE(Mat img)
{
	// path to the trained model
	cv::String model_path = "./src/ImageQuality/model/brisque_model_live.yml";
	// path to range file
	cv::String range_path = "./src/ImageQuality/model/brisque_range_live.yml";
	// ��̬���㷽��
	cv::Scalar result_static = quality::QualityBRISQUE::compute(img, model_path, range_path);
	/* ����һ�ֶ�̬����ķ���
	cv::Ptr<quality::QualityBase> ptr = quality::QualityBRISQUE::create(model_path, range_path);
	// computes BRISQUE score for img
	cv::Scalar result = ptr->compute(img);*/
	return calMEAN(result_static);
}

void qualityCompute(String methodType, Mat img1, Mat img2)
{
	// �㷨������㷨��ʱ
	double result;
	TickMeter costTime;

	costTime.start();
	if ("MSE" == methodType)
		result = MSE(img1, img2);
	else if ("PSNR" == methodType)
		result = PSNR(img1, img2);
	else if ("PSNR" == methodType)
		result = PSNR(img1, img2);
	else if ("GMSD" == methodType)
		result = GMSD(img1, img2);
	else if ("SSIM" == methodType)
		result = SSIM(img1, img2);
	else if ("BRISQUE" == methodType)
		result = BRISQUE(img2);
	costTime.stop();
	cout << methodType << "_result is: " << result << endl;
	cout << methodType << "_cost time is: " << costTime.getTimeSec() / costTime.getCounter() << " s" << endl;
}

int main()
{
	// img1Ϊ��׼ͼ��img2Ϊ���ͼ��
	cv::Mat img1, img2;
	img1 = cv::imread("./src/ImageQuality/image/cut-original-rotated-image.jpg");
	img2 = cv::imread("./src/ImageQuality/image/cut-original-rotated-image.jpg");

	if (img1.empty() || img2.empty())
	{
		cout << "img empty" << endl;
		return 0;
	}

	// ���ԽС�����ͼ��ͻ�׼ͼ��Ĳ��ԽС
	qualityCompute("MSE", img1, img2);
	// ���ԽС�����ͼ��ͻ�׼ͼ��Ĳ��ԽС
	qualityCompute("PSNR", img1, img2);
	// ���Ϊһ��0��1֮�������Խ���ʾ���ͼ��ͻ�׼ͼ��Ĳ��ԽС
	qualityCompute("GMSD", img1, img2);
	// ���Ϊһ��0��1֮�������Խ���ʾ���ͼ��ͻ�׼ͼ��Ĳ��ԽС
	qualityCompute("SSIM", img1, img2);
	// BRISQUE����Ҫ��׼ͼ��
	// ���Ϊһ��0��100֮�������ԽС��ʾ���ͼ������Խ��
	qualityCompute("BRISQUE", cv::Mat{}, img2);
	system("pause");
	return 0;
}
