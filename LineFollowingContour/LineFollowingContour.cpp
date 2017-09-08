/*
 * LineFollowingContour.cpp
 *
 *  Created on: Sep 6, 2017
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

    double upperThreshold = (maxVal - minVal)/2;
    double lowerThreshold = upperThreshold/3;
    printf("[INFO]: Canny edge detection: upper_threshold = %f, lower threshold = %f.\n",
        upperThreshold, lowerThreshold);

    // Do the Canny edge detection.
    Mat edges;
    auto tCannyStart = Clock::now();
    Canny(croppedImgSharpGray, edges, lowerThreshold, upperThreshold);
    auto tCannyEnd = Clock::now();

    printf("[INFO]: Finish the Canny detection in %ld ms.\n",
        chrono::duration_cast<chrono::milliseconds>(tCannyEnd - tCannyStart).count());

    minMaxLoc(edges, &minVal, &maxVal);
    printf("[INFO]: The Canny edge image: minVal = %f, maxVal = %f.\n",
        minVal, maxVal);

    // If the pixels at the four boundary sides are "black" enough, they are considered as
    // part of the black track line and thus part of edges. We have to do this otherwise
    // the contour of the black track line found later won't be closed.
    for (int rowIndex = 0; rowIndex < edges.rows; ++rowIndex)
    {
        if (croppedImgSharpGray.at<uchar>(rowIndex, 0) < minVal + 0.3*(maxVal - minVal))
        {
            edges.at<uchar>(rowIndex, 0) = 255;
        }

        if (croppedImgSharpGray.at<uchar>(rowIndex, edges.cols - 1) < minVal + 0.3*(maxVal - minVal))
        {
            edges.at<uchar>(rowIndex, edges.cols - 1) = 255;
        }
    }
    for (int colIndex = 0; colIndex < edges.cols; ++colIndex)
    {
        if (croppedImgSharpGray.at<uchar>(0, colIndex) < minVal + 0.3*(maxVal - minVal))
        {
            edges.at<uchar>(0, colIndex) = 255;
        }

        if (croppedImgSharpGray.at<uchar>(edges.rows - 1, colIndex) < minVal + 0.3*(maxVal - minVal))
        {
            edges.at<uchar>(edges.rows - 1, colIndex) = 255;
        }
    }

    namedWindow("The raw Canny edges", WINDOW_AUTOSIZE);
    imshow("The raw Canny edges", edges);

    // Find all the contours.
    vector<vector<Point>> contours;
    auto tContourStart = Clock::now();
    findContours(edges, contours, noArray(), RETR_EXTERNAL, CHAIN_APPROX_NONE);
    auto tContourEnd = Clock::now();
    printf("[INFO]: Finish %ld contours in %ld ms.\n", contours.size(),
        chrono::duration_cast<chrono::milliseconds>(tContourEnd - tContourStart).count());

    // Find the contour with the maximum area.
    int maxContourIndex = -1;
    float maxArea = 0;
    for (int contourIndex = 0; contourIndex < static_cast<int>(contours.size()); ++contourIndex)
    {
        float area = contourArea(contours[contourIndex]);
        if (area > maxArea)
        {
            maxContourIndex = contourIndex;
            maxArea = area;
        }
    }

    printf("[INFO]: max contour area = %f.\n", maxArea);

    if (maxContourIndex != -1)
    {
        // Fill the max contour in the source color image. Note that the contour coordinates
        // in srcImg has a down shift of srcImg.rows/2 along the X-axis.
        drawContours(srcImg,
            contours,
            maxContourIndex,
            Scalar(0, 255, 0),
            CV_FILLED,
            8,
            noArray(),
            INT_MAX,
            Point(0, srcImg.rows/2));

        // Compute the X and Y coordinates of the mass center of the max contour. Note that as an
        // approximation, we only count the points at the contour but not the inside points surrounded
        // by the max contour.
        Moments M = moments(contours[maxContourIndex]);
        int cx = static_cast<int>(M.m10/M.m00);
        int cy = static_cast<int>(M.m01/M.m00) + srcImg.rows/2;

        // Draw the lines intersected at the mass center in the source image as well as a filled circle at
        // the mass center.
        line(srcImg, Point(cx, 0), Point(cx, srcImg.rows - 1), Scalar(255, 0, 0));
        line(srcImg, Point(0, cy), Point(srcImg.cols - 1, cy), Scalar(255, 0, 0));
        circle(srcImg, Point(cx, cy), 3, Scalar(255, 0, 0), CV_FILLED);

        namedWindow("Maximum contour", WINDOW_AUTOSIZE);
        imshow("Maximum contour", srcImg);
    }
    else
    {
        printf("[ERROR]: Can't find the contour with the maximum area.\n\n");
    }

    waitKey();
    destroyAllWindows();

    return 0;
}
