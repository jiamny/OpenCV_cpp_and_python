#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	// log_polar_img �����������任���
	// lin_polar_img ������任���
	// recovered_log_polar �������������任���
	// recovered_lin_polar_img ��������任���
	Mat log_polar_img, lin_polar_img, recovered_log_polar, recovered_lin_polar_img;
	// INTER_LINEAR ˫���Բ�ֵ��WARP_FILL_OUTLIERS�������Ŀ��ͼ������
	int flags = INTER_LINEAR + WARP_FILL_OUTLIERS;

	// ��ͼ
	String imagepath = "./src/Polar/image/clock.jpg";
	Mat src = imread(imagepath);
	if (src.empty())
	{
		fprintf(stderr, "Could not initialize capturing...\n");
		return -1;
	}

	// Բ������
	Point2f center((float)src.cols / 2, (float)src.rows / 2);
	// Բ�İ뾶
	double maxRadius = min(center.y, center.x);

	// direct transform
	// linear Polar ������任, Size()��ʾOpenCV�����������о������ͼ��ߴ�
	warpPolar(src, lin_polar_img, Size(), center, maxRadius, flags);
	// semilog Polar �����������任, Size()��ʾOpenCV�����������о������ͼ��ߴ�
	warpPolar(src, log_polar_img, Size(), center, maxRadius, flags + WARP_POLAR_LOG);
	// inverse transform ��任
	warpPolar(lin_polar_img, recovered_lin_polar_img, src.size(), center, maxRadius, flags + WARP_INVERSE_MAP);
	warpPolar(log_polar_img, recovered_log_polar, src.size(), center, maxRadius, flags + WARP_POLAR_LOG + WARP_INVERSE_MAP);

	// �ı�������
	// rotate(lin_polar_img, lin_polar_img, ROTATE_90_CLOCKWISE);

	// չʾͼƬ
	imshow("Src frame", src);
	imshow("Log-Polar", log_polar_img);
	imshow("Linear-Polar", lin_polar_img);
	imshow("Recovered Linear-Polar", recovered_lin_polar_img);
	imshow("Recovered Log-Polar", recovered_log_polar);
	waitKey(0);
	system("pause");
	return 0;
}
