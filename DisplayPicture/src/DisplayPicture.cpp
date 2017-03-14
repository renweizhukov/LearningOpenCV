// Modified from Example 2-2 in the book "Learning OpenCV 3: Computer Vision in C++ with the OpenCV Library".

#include <opencv2/highgui.hpp>

using namespace cv;

int main(int argc, char** argv)
{
	Mat img = imread(argv[1], -1);

	if (img.empty())
	{
		return -1;
	}

	namedWindow("Example 2-2", WINDOW_AUTOSIZE);
	imshow("Example 2-2", img);
	waitKey(0);

	destroyWindow("Example 2-2");
}
