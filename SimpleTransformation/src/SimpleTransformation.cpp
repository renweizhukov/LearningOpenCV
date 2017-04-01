// Modified from Example 2-5 in the book "Learning OpenCV 3: Computer Vision in C++ with the OpenCV Library".

#include <cstdio>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  // Load an image specified on the command line.
  //
  Mat image_in = imread(argv[1],-1);

  // Create some windows to show the input
  // and output images in.
  //
  namedWindow("Example 2-5-in", WINDOW_AUTOSIZE);
  namedWindow("Example 2-5-out", WINDOW_AUTOSIZE);

  // Create a window to show our input image
  //
  imshow("Example 2-5-in", image_in);

  // Create an image to hold the smoothed output
  //
  Mat image_out;

  // Do the smoothing
  // ( Note: Could use GaussianBlur(), blur(), medianBlur() or
  // bilateralFilter(). )
  //
  GaussianBlur(image_in, image_out, Size(5,5), 3, 3);

  // Show the smoothed image in the output window
  //
  imshow("Example 2-5-out", image_out);

  // Save the image
  //
  string fileName(argv[1]);
  size_t dotPos = fileName.find_last_of('.');
  if (dotPos != string::npos)
  {
	  fileName = fileName.substr(0, dotPos) + "-new" + fileName.substr(dotPos);
  }
  else
  {
	  // The file name doesn't contain an extension.
	  fileName += ".jpg";
  }
  printf("Saving the blurred image to %s.\n", fileName.c_str());
  try{
	  imwrite(fileName, image_out);
  }
  catch (runtime_error& ex)
  {
	  printf("Exception converting image to PNG format: %s\n", ex.what());
	  return 1;
  }

  printf("Saved the blurred image to %s.\n", fileName.c_str());

  // Wait for the user to hit a key, windows will self destruct
  //
  waitKey( 0 );

  // Destroy the two windows.
  //
  destroyWindow("Example 2-5-in");
  destroyWindow("Example 2-5-out");
}
