#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char *argv[]) {

    Mat markerImage;
    Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    aruco::drawMarker(dictionary, 33, 200, markerImage, 1);

    imwrite("./src/AugmentedRealityWithArucoMarkers/marker33.png", markerImage);
    std::cout << "Done\n";
}
