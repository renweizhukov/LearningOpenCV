// Modified from Example 2-3 in the book "Learning OpenCV 3: Computer Vision in C++ with the OpenCV Library".

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;

int main(int argc, char** argv)
{
	namedWindow("Example 2-3", WINDOW_AUTOSIZE);
	VideoCapture cap;
	cap.open(String(argv[1]));

	Mat frame;
	for(;;)
	{
		cap >> frame;
		if (frame.empty())
		{
			break;
		}

		imshow("Example 2-3", frame);
		if (waitKey(33) >= 0)
		{
			break;
		}
	}

	return 0;
}
