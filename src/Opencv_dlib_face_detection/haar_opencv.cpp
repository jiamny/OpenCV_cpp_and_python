#include "face_detection.h"

/**
 * @brief 人脸检测haar级联
 *
 * @param frame 原图
 * @param faceCascadePath 模型文件
 * @return Mat
 */
Mat detectFaceHaar(Mat frame, string faceCascadePath)
{
	//图像缩放
	auto inHeight = 300;
	auto inWidth = 0;
	if (!inWidth)
	{
		inWidth = (int)(((float)frame.cols / (float)frame.rows) * inHeight);
	}
	resize(frame, frame, Size(inWidth, inHeight));

	//转换为灰度图
	Mat frameGray = frame.clone();
	//cvtColor(frame, frameGray, CV_BGR2GRAY);

	//级联分类器
	CascadeClassifier faceCascade;
	faceCascade.load(faceCascadePath);
	std::vector<Rect> faces;
	faceCascade.detectMultiScale(frameGray, faces);

	for (size_t i = 0; i < faces.size(); i++)
	{
		int x1 = faces[i].x;
		int y1 = faces[i].y;
		int x2 = faces[i].x + faces[i].width;
		int y2 = faces[i].y + faces[i].height;
		Rect face_rect(Point2i(x1, y1), Point2i(x2, y2));
		rectangle(frameGray, face_rect, cv::Scalar(0, 255, 0), 2, 4);
	}
	return frameGray;
}



