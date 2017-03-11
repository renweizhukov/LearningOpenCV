#include <opencv2/opencv.hpp>
using namespace cv;

int main ( int argc, char **argv )
{
    Mat img(480, 640, CV_8U);
    putText(img, "Hello World!", Point( 200, 400 ), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 1.0, Scalar( 255, 255, 0 ));
    imshow("My Window", img);
    waitKey();
    return 0;
}
