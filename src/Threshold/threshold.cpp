/**
 *  OpenCV Threshold Example
 *   
 *  Copyright 2015 by Satya Mallick <spmallick@gmail.com>
 *  
 **/

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{

	// Read image 
	Mat src = imread("./src/Threshold/threshold.png", IMREAD_GRAYSCALE);
	Mat dst; 
	
	// Basic threhold example 
	threshold(src,dst,0, 255, THRESH_BINARY); 
	imwrite("./src/Threshold/opencv-threshold-example.jpg", dst);

	// Thresholding with maxval set to 128
	threshold(src, dst, 0, 128, THRESH_BINARY); 
	imwrite("./src/Threshold/opencv-thresh-binary-maxval.jpg", dst);
	
	// Thresholding with threshold value set 127 
	threshold(src,dst,127,255, THRESH_BINARY); 
	imwrite("./src/Threshold/opencv-thresh-binary.jpg", dst);
	
	// Thresholding using THRESH_BINARY_INV 
	threshold(src,dst,127,255, THRESH_BINARY_INV); 
	imwrite("./src/Threshold/opencv-thresh-binary-inv.jpg", dst);
	
	// Thresholding using THRESH_TRUNC 
	threshold(src,dst,127,255, THRESH_TRUNC); 
	imwrite("./src/Threshold/opencv-thresh-trunc.jpg", dst);

	// Thresholding using THRESH_TOZERO 
	threshold(src,dst,127,255, THRESH_TOZERO); 
	imwrite("./src/Threshold/opencv-thresh-tozero.jpg", dst);

	// Thresholding using THRESH_TOZERO_INV 
	threshold(src,dst,127,255, THRESH_TOZERO_INV); 
	imwrite("./src/Threshold/opencv-thresh-to-zero-inv.jpg", dst);
} 
