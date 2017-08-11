/*
 * FindMaxConvexContour.cpp
 *
 *  Created on: Aug 8, 2017
 *      Author: renwei
 */

#include <iostream>
#include <vector>
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
        cerr << "[ERROR]: No source image is specified." << endl;
        return -1;
    }

    Mat srcImg = imread(argv[1]);
    if (srcImg.empty())
    {
        cerr << "[ERROR]: Can't load the image file " << argv[1] << "." << endl;
        return -1;
    }

    // Display the original color image.
    namedWindow("The original color image", WINDOW_AUTOSIZE);
    imshow("The original color image", srcImg);

    // Sharpen the image using Gaussian Blur. Note that srcImgGaussian = 1.5*srcImg - 0.5*srcImgGaussian,
    // but to avoid overflow while multiplying srcImg by 1.5, we subtract srcImgGaussian from srcImg first
    // and then add 0.5*srcImg.
    Mat srcImgGaussian;
    GaussianBlur(srcImg, srcImgGaussian, Size(0,0), 3);
    addWeighted(srcImg, 1.0, srcImgGaussian, -0.5, 0.0, srcImgGaussian);
    addWeighted(srcImg, 0.5, srcImgGaussian, 1.0, 0.0, srcImgGaussian);

    namedWindow("The image after Gaussian Blurring", WINDOW_AUTOSIZE);
    imshow("The image after Gaussian Blurring", srcImgGaussian);

    // Convert the color image after Gaussian Blur into grayscale.
    Mat srcImgGaussianGray;
    cvtColor(srcImgGaussian, srcImgGaussianGray, COLOR_BGR2GRAY);

    // Display the grayscale image after Gaussian Blur.
    namedWindow("The grayscale image after Gaussian Blur", WINDOW_AUTOSIZE);
    imshow("The grayscale image after Gaussian Blur", srcImgGaussianGray);

    // Enhance the contrast by doing the histogram equalization.
    Mat srcImgGaussianGrayEqualized;
    equalizeHist(srcImgGaussianGray, srcImgGaussianGrayEqualized);

    // Display the equalized grayscale image.
    namedWindow("The equalized grayscale image after Gaussian Blur", WINDOW_AUTOSIZE);
    imshow("The equalized grayscale image after Gaussian Blur", srcImgGaussianGrayEqualized);

    // Compute the mean and standard deviation to get the estimated lower and upper threshold
    // for the Canny edge detection.
    Mat meanMat;
    Mat stdDevMat;
    meanStdDev(srcImgGaussianGrayEqualized, meanMat, stdDevMat);

    double mean = meanMat.at<double>(0, 0);
    double stdDev = stdDevMat.at<double>(0, 0);
    cout << "[INFO]: Mean of the equalized grayscale image = " << mean << "." << endl;
    cout << "[INFO]: Standard deviation of the equalized grayscale image = " << stdDev << "." << endl;

    double upperThreshold = min(mean, 255.0);
    double lowerThreshold = upperThreshold/3.0;
    cout << "[INFO]: Canny edge detection: lower_threshold = " << lowerThreshold
        << ", upper_threshold = " << upperThreshold << "." << endl;

    // Do the Canny edge detection.
    Mat edges;
    auto tCannyStart = Clock::now();
    Canny(srcImgGaussianGrayEqualized, edges, lowerThreshold, upperThreshold);
    auto tCannyEnd = Clock::now();
    cout << "[INFO]: Finish the Canny detection in "
        << chrono::duration_cast<chrono::milliseconds>(tCannyEnd - tCannyStart).count() << " milliseconds." << endl;

    namedWindow("The raw Canny edges", WINDOW_AUTOSIZE);
    imshow("The raw Canny edges", edges);

    // Find lines using the progressive probabilistic Hough Transform.
    vector<Vec4i> linesEndpoints;
    const double rhoResolution = 1.0;
    const double thetaResolution = CV_PI/180.0;
    const int houghThreshold = 5;
    const double minLineLen = 100;
    const double maxLineGap = 10;
    auto tHoughStart = Clock::now();
    HoughLinesP(
        edges,
        linesEndpoints,
        rhoResolution,
        thetaResolution,
        houghThreshold,
        minLineLen,
        maxLineGap);
    auto tHoughEnd = Clock::now();
    cout << "[INFO]: Finish the Hough Transform in "
        << chrono::duration_cast<chrono::milliseconds>(tHoughEnd - tHoughStart).count() << " milliseconds." << endl;

    // Overlay the Hough lines on the Canny edges. Also add two "virtual" lines to
    // denote the left and right boundaries of the object, e.g., a book.
    int leftBoundaryX = 0;
    int leftBoundaryY = 0;
    int rightBoundaryX = edges.cols - 1;
    int rightBoundaryY = 0;

    Mat lines(edges.size(), CV_8UC1, Scalar::all(0));

    for (const auto& lineEndpoints : linesEndpoints)
    {
        if (lineEndpoints[0] < maxLineGap)
        {
            if (lineEndpoints[1] > leftBoundaryY)
            {
                leftBoundaryX = lineEndpoints[0];
                leftBoundaryY = lineEndpoints[1];
            }
        }
        else if (lineEndpoints[0] >= edges.cols - maxLineGap)
        {
            if (lineEndpoints[1] > rightBoundaryY)
            {
                rightBoundaryX = lineEndpoints[0];
                rightBoundaryY = lineEndpoints[1];
            }
        }

        if (lineEndpoints[2] < maxLineGap)
        {
            if (lineEndpoints[3] > leftBoundaryY)
            {
                leftBoundaryX = lineEndpoints[2];
                leftBoundaryY = lineEndpoints[3];
            }
        }
        else if (lineEndpoints[2] >= edges.cols - maxLineGap)
        {
            if (lineEndpoints[3] > rightBoundaryY)
            {
                rightBoundaryX = lineEndpoints[2];
                rightBoundaryY = lineEndpoints[3];
            }
        }

        Point pt1(lineEndpoints[0], lineEndpoints[1]);
        Point pt2(lineEndpoints[2], lineEndpoints[3]);

        // Draw the white hough lines in a totally black image.
        line(lines, pt1, pt2, Scalar(255, 255, 255));

        // Overlay the hough lines on the Canny edges.
        line(edges, pt1, pt2, Scalar(255, 255, 255));
    }

    cout << "[INFO]: leftBoundaryX = " << leftBoundaryX << ", leftBoundaryY = " << leftBoundaryY << "." << endl;
    cout << "[INFO]: rightBoundaryX = " << rightBoundaryX << ", rightBoundaryY = " << rightBoundaryY << "." << endl;

    // Set the left and right boundaries of the object.
    for (int rowIndex = leftBoundaryY; rowIndex < edges.rows; ++rowIndex)
    {
        edges.at<char>(rowIndex, leftBoundaryX) = 255;
    }
    for (int rowIndex = rightBoundaryY; rowIndex < edges.rows; ++rowIndex)
    {
        edges.at<char>(rowIndex, rightBoundaryX) = 255;
    }

    // Set the lower boundary of the book.
    for (int colIndex = 0; colIndex < edges.cols; ++colIndex)
    {
        edges.at<char>(edges.rows - 1, colIndex) = 255;
    }

    // Display the hough lines.
    namedWindow("The Hough lines", WINDOW_AUTOSIZE);
    imshow("The Hough lines", lines);

    // Display the Canny edges with the Hough lines.
    namedWindow("The Canny edges with Hough lines", WINDOW_AUTOSIZE);
    imshow("The Canny edges with Hough lines", edges);

    // Find all the contours.
    vector<vector<Point>> contours;
    auto tContourStart = Clock::now();
    findContours(edges, contours, noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    auto tContourEnd = Clock::now();
    cout << "[INFO]: Finish " << contours.size() << " contours in "
        << chrono::duration_cast<chrono::milliseconds>(tContourEnd - tContourStart).count() << " milliseconds." << endl;

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

    cout << "[INFO]: max contour area = " << maxArea << "." << endl;

    if (maxContourIndex != -1)
    {
        // Get the convex hull of the max contour.
        vector<Point> hull;
        convexHull(contours[maxContourIndex], hull);

        float maxConvexArea = contourArea(hull);
        cout << "[INFO]: max convex contour area = " << maxConvexArea << "." << endl;

        // Draw the max contour and its convex hull in the original color image.
        drawContours(srcImg, contours, maxContourIndex, Scalar::all(255));
        drawContours(srcImg, vector<vector<Point>>{hull}, 0, Scalar(0, 0, 255));

        namedWindow("Maximum contour and its convex hull", WINDOW_AUTOSIZE);
        imshow("Maximum contour and its convex hull", srcImg);
    }
    else
    {
        cerr << "[ERROR]: Can't find the contour with the maximum area." << endl;
    }

    waitKey();
    destroyAllWindows();

    return 0;
}
