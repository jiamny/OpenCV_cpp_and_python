#include <opencv2/photo.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
using namespace cv;
using namespace std;


// Read img files from the directory
void readImages(string dirName, vector<Mat> &images)
{

  cout << "Reading images from " << dirName;

  // Add slash to directory name if missing
  if (!dirName.empty() && dirName.back() != '/')
    dirName += '/';

  DIR *dir;
  struct dirent *ent;
  int count = 0;

  //image extensions
  string imgExt = "jpg";
  vector<string> files;

  if ((dir = opendir (dirName.c_str())) != NULL)
  {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL)
    {
      if(strcmp(ent->d_name,".") == 0 || strcmp(ent->d_name,"..") == 0 )
      {
        continue;
      }
      string fname = ent->d_name;

      if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos) {
        string path = dirName + fname;
        Mat img = imread(path);
        if(!img.data) {
          cout << "image " << path << " not read properly" << endl;
        } else {
          images.push_back(img);
        }
      }
    }
    closedir (dir);
  }

  // Exit program if no images are found
  if(images.empty())exit(EXIT_FAILURE);

  cout << "... " << images.size() / 2 << " files read"<< endl;

}


int main(int argc, char **argv)
{
  // Read images
  cout << "Reading images ... " << endl;
  vector<Mat> images;
  
  bool needsAlignment = true;
  if(argc > 1) {
    // Read images from the command line
    for(int i=1; i < argc; i++)
    {
      Mat im = imread(argv[i]);
      images.push_back(im);
    }

  } else {
    // Read example images
	 readImages( "./src/ExposureFusion/images/", images);
     needsAlignment = false;
  }
  std::cout << "img files: " << images.size() << std::endl;
  
  // Align input images
  if(needsAlignment)
  {
    cout << "Aligning images ... " << endl;
    Ptr<AlignMTB> alignMTB = createAlignMTB();
    alignMTB->process(images, images);
  } else {
    cout << "Skipping alignment ... " << endl;
  }

  // Merge using Exposure Fusion
  cout << "Merging using Exposure Fusion ... " << endl;
  Mat exposureFusion;
  Ptr<MergeMertens> mergeMertens = createMergeMertens();
  mergeMertens->process(images, exposureFusion);
  
  cv::imshow("exposureFusion", exposureFusion);
  cv::waitKey(0);
  // Save output image
  cout << "Saving output ... exposure-fusion.jpg"<< endl;
  imwrite("./src/ExposureFusion/exposure-fusion.jpg", exposureFusion * 255);
  
  return EXIT_SUCCESS;
}
