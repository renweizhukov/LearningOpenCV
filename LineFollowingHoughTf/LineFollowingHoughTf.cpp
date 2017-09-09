/*
 * LineFollowingHoughTf.cpp
 *
 *  Created on: Sep 8, 2017
 *      Author: renwei
 */

#include <cstdio>
#include <algorithm>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("[ERROR]: No source image is specified.\n\n");
        return -1;
    }

    Mat srcImg = imread(argv[1]);
    if (srcImg.empty())
    {
        printf("[ERROR]: Can't load the image file %s.\n\n", argv[1]);
        return -1;
    }

    // Crop the image to the bottom half of the source image so that the line which may be farther down
    // the track won't interfere with the line following. Note that croppedImg is not a copy of srcImg
    // and it shares the same storage as srcImg.
    Mat croppedImg = srcImg(Range(srcImg.rows/2, srcImg.rows), Range(0, srcImg.cols));

    // Sharpen the image using Unsharp Masking with a Gaussian blurred version of the image. Note that
    // srcImgGaussian = 1.5*srcImg - 0.5*srcImgGaussian, but to avoid overflow while multiplying srcImg
    // by 1.5, we subtract srcImgGaussian from srcImg first and then add 0.5*srcImg.
    Mat croppedImgSharp;
    GaussianBlur(croppedImg, croppedImgSharp, Size(0, 0), 3);
    addWeighted(croppedImg, 1.0, croppedImgSharp, -0.5, 0.0, croppedImgSharp);
    addWeighted(croppedImg, 0.5, croppedImgSharp, 1.0, 0.0, croppedImgSharp);

    namedWindow("The image after Unsharp Masking", WINDOW_AUTOSIZE);
    imshow("The image after Unsharp Masking", croppedImgSharp);

    // Convert the color image after Gaussian Blur into grayscale.
    Mat croppedImgSharpGray;
    cvtColor(croppedImgSharp, croppedImgSharpGray, COLOR_BGR2GRAY);

    // Display the grayscale image after Unsharp Masking.
    namedWindow("The grayscale image after Unsharp Masking", WINDOW_AUTOSIZE);
    imshow("The grayscale image after Unsharp Masking", croppedImgSharpGray);

    double minVal = 0.0;
    double maxVal = 0.0;
    minMaxLoc(croppedImgSharpGray, &minVal, &maxVal);
    printf("[INFO]: The grayscale image after Unsharp Masking: minVal = %f, maxVal = %f.\n",
        minVal, maxVal);

    double thresh = minVal + 0.3*(maxVal - minVal);
    Mat croppedImgThresholded;
    threshold(croppedImgSharpGray, croppedImgThresholded, thresh, 255, THRESH_BINARY_INV);

    // Display the thresholded image.
    namedWindow("The thresholded image", WINDOW_AUTOSIZE);
    imshow("The thresholded image", croppedImgThresholded);

    // Find lines using the progressive probabilistic Hough Transform.
    vector<Vec4i> linesEndpoints;
    const double rhoResolution = 1.0;
    const double thetaResolution = CV_PI/180.0;
    const int houghThreshold = 5;
    const double minLineLen = 25;
    const double maxLineGap = 5;
    auto tHoughStart = Clock::now();
    HoughLinesP(
        croppedImgThresholded,
        linesEndpoints,
        rhoResolution,
        thetaResolution,
        houghThreshold,
        minLineLen,
        maxLineGap);
    auto tHoughEnd = Clock::now();
    printf("[INFO]: Finish the Hough Transform in %ld ms.\n",
        chrono::duration_cast<chrono::milliseconds>(tHoughEnd - tHoughStart).count());

    // Display only the Hough lines and overlay the Hough lines on the original source color image.
    Mat lines(croppedImgThresholded.size(), CV_8UC1, Scalar::all(0));

    for (const auto& lineEndpoints : linesEndpoints)
    {
        Point pt1(lineEndpoints[0], lineEndpoints[1]);
        Point pt2(lineEndpoints[2], lineEndpoints[3]);

        // Draw the white hough lines in a totally black image.
        line(lines, pt1, pt2, Scalar(255, 255, 255));

        // Overlay the hough lines on the Canny edges.
        //line(srcImg,
        //    Point(pt1.x, pt1.y + srcImg.rows/2),
        //    Point(pt2.x, pt2.y + srcImg.rows/2),
        //    Scalar(0, 0, 255));
    }

    // Display the hough lines.
    namedWindow("The Hough lines", WINDOW_AUTOSIZE);
    imshow("The Hough lines", lines);

    // Find all the contours.
    vector<vector<Point>> contours;
    auto tContourStart = Clock::now();
    findContours(lines, contours, noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    auto tContourEnd = Clock::now();
    printf("[INFO]: Finish %ld contours in %ld ms.\n", contours.size(),
        chrono::duration_cast<chrono::milliseconds>(tContourEnd - tContourStart).count());

    // Find the contour with the maximum length.
    int maxContourIndex = -1;
    float maxLen = 0;
    for (int contourIndex = 0; contourIndex < static_cast<int>(contours.size()); ++contourIndex)
    {
        float len = arcLength(contours[contourIndex], false);
        if (len > maxLen)
        {
            maxContourIndex = contourIndex;
            maxLen = len;
        }
    }

    printf("[INFO]: max contour length = %f.\n", maxLen);

    if (maxContourIndex != -1)
    {
        // Draw the max contour in the original color image.
        drawContours(srcImg,
            contours,
            maxContourIndex,
            Scalar(0, 255, 0),
            CV_FILLED,
            8,
            noArray(),
            INT_MAX,
            Point(0, srcImg.rows/2));

        namedWindow("Maximum contour and its convex hull", WINDOW_AUTOSIZE);
        imshow("Maximum contour and its convex hull", srcImg);
    }
    else
    {
        printf("[ERROR]: Can't find the contour with the maximum area.\n\n");
    }

    waitKey();
    destroyAllWindows();

    return 0;
}
