// ��ͬͼ�񳬷��㷨Ч������

#include <iostream>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/quality.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

// չʾͼƬ
static void showBenchmark(vector<Mat> images, string title, Size imageSize,
	const vector<String> imageTitles,
	const vector<double> psnrValues,
	const vector<double> ssimValues)
{
	// ������Ϣ
	int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
	int fontScale = 1;
	Scalar fontColor = Scalar(255, 255, 255);

	// ͼ������
	int len = static_cast<int>(images.size());

	int cols = 2, rows = 2;

	// ��������ͼ��
	Mat fullImage = Mat::zeros(Size((cols * 10) + imageSize.width * cols, (rows * 10) + imageSize.height * rows),
		images[0].type());

	stringstream ss;
	int h_ = -1;
	// ƴ����ʾͼƬ
	for (int i = 0; i < len; i++)
	{
		int fontStart = 15;
		int w_ = i % cols;
		if (i % cols == 0)
			h_++;

		Rect ROI((w_ * (10 + imageSize.width)), (h_ * (10 + imageSize.height)), imageSize.width, imageSize.height);
		Mat tmp;
		resize(images[i], tmp, Size(ROI.width, ROI.height));

		ss << imageTitles[i];
		putText(tmp,
			ss.str(),
			Point(5, fontStart),
			fontFace,
			fontScale,
			fontColor,
			1,
			16);

		ss.str("");
		fontStart += 20;

		ss << "PSNR: " << psnrValues[i];
		putText(tmp,
			ss.str(),
			Point(5, fontStart),
			fontFace,
			fontScale,
			fontColor,
			1,
			16);

		ss.str("");
		fontStart += 20;

		ss << "SSIM: " << ssimValues[i];
		putText(tmp,
			ss.str(),
			Point(5, fontStart),
			fontFace,
			fontScale,
			fontColor,
			1,
			16);

		ss.str("");
		fontStart += 20;

		tmp.copyTo(fullImage(ROI));
	}

	namedWindow(title, 1);
	imshow(title, fullImage);
	imwrite("./src/ImageSuperres/save.jpg", fullImage);
	waitKey();
}

static Vec2d getQualityValues(Mat orig, Mat upsampled)
{
	double psnr = PSNR(upsampled, orig);
	// ǰ��������Ϊ�Ա�ͼƬ������������Ϊ�������
	Scalar q = quality::QualitySSIM::compute(upsampled, orig, noArray());
	double ssim = mean(Vec3d((q[0]), q[1], q[2]))[0];
	return Vec2d(psnr, ssim);
}

int main()
{
	// ͼƬ·��
	string img_path = string("./src/ImageSuperres/image/image.png");
	// �㷨���� edsr, espcn, fsrcnn or lapsrn
	string algorithm = string("lapsrn");

	// ģ��·���������㷨ȷ��
	string model = string("./models/ImageSuperres/LapSRN_x2.pb");
	// �Ŵ�ϵ��
	int scale = 2;

	Mat img = imread(img_path);
	if (img.empty())
	{
		cerr << "Couldn't load image: " << img_path << "\n";
		return -2;
	}

	// Crop the image so the images will be aligned
	// �ü�ͼ��ʹͼ�����
	int width = img.cols - (img.cols % scale);
	int height = img.rows - (img.rows % scale);
	Mat cropped = img(Rect(0, 0, width, height));

	// Downscale the image for benchmarking
	// ��Сͼ����ʵ�ֻ�׼��������
	Mat img_downscaled;
	resize(cropped, img_downscaled, Size(), 1.0 / scale, 1.0 / scale);

	// Make dnn super resolution instance
	// ����ģ�ͳ�ʼ��
	DnnSuperResImpl sr;

	vector<Mat> allImages;
	// �Ŵ���ͼƬ
	Mat img_new;

	// Read and set the dnn model
	// ��ȡ���趨ģ��
	sr.readModel(model);
	sr.setModel(algorithm, scale);
	// �Ŵ�ͼ��
	sr.upsample(img_downscaled, img_new);

	vector<double> psnrValues = vector<double>();
	vector<double> ssimValues = vector<double>();

	// DL MODEL
	// ���ģ����������ֵ
	Vec2f quality = getQualityValues(cropped, img_new);

	// ģ����������PSNR
	psnrValues.push_back(quality[0]);
	// ģ����������SSIM
	ssimValues.push_back(quality[1]);

	// ��ֵԽ��ͼ������Խ��
	cout << sr.getAlgorithm() << ":" << endl;
	cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
	cout << "----------------------" << endl;

	// BICUBIC
	// INTER_CUBIC - ����������ֵ�Ŵ�ͼ��
	Mat bicubic;
	resize(img_downscaled, bicubic, Size(), scale, scale, INTER_CUBIC);
	quality = getQualityValues(cropped, bicubic);

	psnrValues.push_back(quality[0]);
	ssimValues.push_back(quality[1]);

	cout << "Bicubic " << endl;
	cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
	cout << "----------------------" << endl;

	// NEAREST NEIGHBOR
	// INTER_NEAREST - ����ڲ�ֵ
	Mat nearest;
	resize(img_downscaled, nearest, Size(), scale, scale, INTER_NEAREST);
	quality = getQualityValues(cropped, nearest);

	psnrValues.push_back(quality[0]);
	ssimValues.push_back(quality[1]);

	cout << "Nearest neighbor" << endl;
	cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
	cout << "----------------------" << endl;

	// LANCZOS
	// Lanczos��ֵ�Ŵ�ͼ��
	Mat lanczos;
	resize(img_downscaled, lanczos, Size(), scale, scale, INTER_LANCZOS4);
	quality = getQualityValues(cropped, lanczos);

	psnrValues.push_back(quality[0]);
	ssimValues.push_back(quality[1]);

	cout << "Lanczos" << endl;
	cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
	cout << "-----------------------------------------------" << endl;

	// Ҫ��ʾ��ͼƬ
	vector<Mat> imgs{ img_new, bicubic, nearest, lanczos };
	// Ҫ��ʾ�ı���
	vector<String> titles{ sr.getAlgorithm(), "Bicubic", "Nearest neighbor", "Lanczos" };
	showBenchmark(imgs, "Quality benchmark", Size(bicubic.cols, bicubic.rows), titles, psnrValues, ssimValues);

	waitKey(0);

	return 0;
}
