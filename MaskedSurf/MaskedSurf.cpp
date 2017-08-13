/*
 * MaskedSurf.cpp
 *
 *  Created on: Aug 11, 2017
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
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

typedef std::chrono::high_resolution_clock Clock;

bool GetSurfMask(const Mat& maskImg, Mat& mask, vector<Point>& maxConvexContour)
{
    Mat img;
    maskImg.copyTo(img);

    // Sharpen the image using Gaussian Blur. Note that imgGaussian = 1.5*img - 0.5*imgGaussian,
    // but to avoid overflow while multiplying img by 1.5, we subtract imgGaussian from img first
    // and then add 0.5*img.
    Mat imgGaussian;
    GaussianBlur(img, imgGaussian, Size(0,0), 3);
    addWeighted(img, 1.0, imgGaussian, -0.5, 0.0, imgGaussian);
    addWeighted(img, 0.5, imgGaussian, 1.0, 0.0, imgGaussian);

    // Convert the color image after Gaussian Blur into grayscale.
    Mat imgGaussianGray;
    cvtColor(imgGaussian, imgGaussianGray, COLOR_BGR2GRAY);

    // Enhance the contrast by doing the histogram equalization.
    Mat imgGaussianGrayEqualized;
    equalizeHist(imgGaussianGray, imgGaussianGrayEqualized);

    // Compute the mean and standard deviation to get the estimated lower and upper threshold
    // for the Canny edge detection.
    Mat meanMat;
    Mat stdDevMat;
    meanStdDev(imgGaussianGrayEqualized, meanMat, stdDevMat);

    double mean = meanMat.at<double>(0, 0);
    double stdDev = stdDevMat.at<double>(0, 0);
    cout << "[DEBUG]: Mean of the equalized grayscale image = " << mean << "." << endl;
    cout << "[DEBUG]: Standard deviation of the equalized grayscale image = " << stdDev << "." << endl;

    double upperThreshold = min(mean, 255.0);
    double lowerThreshold = upperThreshold/3.0;
    cout << "[DEBUG]: Canny edge detection: lower_threshold = " << lowerThreshold
        << ", upper_threshold = " << upperThreshold << "." << endl;

    // Do the Canny edge detection.
    Mat edges;
    auto tCannyStart = Clock::now();
    Canny(imgGaussianGrayEqualized, edges, lowerThreshold, upperThreshold);
    auto tCannyEnd = Clock::now();
    cout << "[INFO]: Finish the Canny detection in "
        << chrono::duration_cast<chrono::milliseconds>(tCannyEnd - tCannyStart).count() << " ms." << endl;

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
    cout << "[DEBUG]: Finish the Hough Transform in "
        << chrono::duration_cast<chrono::milliseconds>(tHoughEnd - tHoughStart).count() << " ms." << endl;

    // Overlay the Hough lines on the Canny edges. Also add two "virtual" lines to
    // denote the left and right boundaries of the object, e.g., a book.
    int leftBoundaryX = 0;
    int leftBoundaryY = 0;
    int rightBoundaryX = edges.cols - 1;
    int rightBoundaryY = 0;

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

        // Overlay the hough lines on the Canny edges.
        line(edges, pt1, pt2, Scalar(255, 255, 255));
    }

    cout << "[DEBUG]: leftBoundaryX = " << leftBoundaryX << ", leftBoundaryY = " << leftBoundaryY << "." << endl;
    cout << "[DEBUG]: rightBoundaryX = " << rightBoundaryX << ", rightBoundaryY = " << rightBoundaryY << "." << endl;

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

    // Find all the contours.
    vector<vector<Point>> contours;
    auto tContourStart = Clock::now();
    findContours(edges, contours, noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    auto tContourEnd = Clock::now();
    cout << "[DEBUG]: Finish " << contours.size() << " contours in "
        << chrono::duration_cast<chrono::milliseconds>(tContourEnd - tContourStart).count() << " ms." << endl;

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

    cout << "[DEBUG]: max contour area = " << maxArea << "." << endl;

    mask = Mat::zeros(maskImg.size(), CV_8UC1);
    if (maxContourIndex != -1)
    {
        // Get the convex hull of the max contour.
        convexHull(contours[maxContourIndex], maxConvexContour);

        float maxConvexArea = contourArea(maxConvexContour);
        cout << "[INFO]: mask area = " << maxConvexArea << "." << endl;

        // Draw the max convex contour in the totally black image to create the mask.
        drawContours(mask, vector<vector<Point>>{maxConvexContour}, 0, Scalar(255), CV_FILLED);

        return true;
    }
    else
    {
        cerr << "[ERROR]: Can't find the contour with the maximum area." << endl << endl;
        return false;
    }
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cerr << "[ERROR]: Either the source image or the mask image or both are not specified." << endl << endl;
        return -1;
    }

    Mat srcImg = imread(argv[1]);
    if (srcImg.empty())
    {
        cerr << "[ERROR]: Can't load the source image file " << argv[1] << "." << endl << endl;
        return -1;
    }

    Mat maskImg = imread(argv[2]);
    if (maskImg.empty())
    {
        cerr << "[ERROR]: Can't load the maskimage file " << argv[2] << "." << endl << endl;
        return -1;
    }

    // Detect the keypoints using the SURF detector and compute the descriptors with no mask.
    const int minHessian = 400;
    Ptr<SURF> surfDetector = SURF::create(minHessian);

    vector<KeyPoint> keypointsWithNoMask;
    Mat descriptorsWithNoMask;

    auto tStart = Clock::now();
    surfDetector->detectAndCompute(srcImg, noArray(), keypointsWithNoMask, descriptorsWithNoMask);
    auto tEnd = Clock::now();
    cout << "[INFO]: Detect " << keypointsWithNoMask.size() << " keypoints with NO Mask in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count() << " ms." << endl;

    // Draw the keypoints with no mask in the source image.
    Mat srcImgAndKeypointsWithNoMask;
    drawKeypoints(srcImg, keypointsWithNoMask, srcImgAndKeypointsWithNoMask);

    namedWindow("Keypoints with NO Mask", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    imshow("Keypoints with NO Mask", srcImgAndKeypointsWithNoMask);

    // Compute a mask from the mask image.
    Mat mask;
    vector<Point> maxConvexContour;
    bool bRet = GetSurfMask(maskImg, mask, maxConvexContour);

    if (bRet)
    {
        vector<KeyPoint> keypointsWithMask;
        Mat descriptorsWithMask;

        tStart = Clock::now();
        surfDetector->detectAndCompute(srcImg, mask, keypointsWithMask, descriptorsWithMask);
        tEnd = Clock::now();
        cout << "[INFO]: Detect " << keypointsWithMask.size() << " keypoints with Mask in "
            << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count() << " ms." << endl;

        // Draw the keypoints with mask in the source image.
        Mat srcImgAndKeypointsWithMask;
        drawKeypoints(srcImg, keypointsWithMask, srcImgAndKeypointsWithMask);

        // Draw the max convex contour in the source image.
        drawContours(srcImgAndKeypointsWithMask, vector<vector<Point>>{maxConvexContour}, 0, Scalar(0, 0, 255), 2);

        namedWindow("Keypoints with Mask", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
        imshow("Keypoints with Mask", srcImgAndKeypointsWithMask);
    }
    else
    {
        cerr << "[ERROR]: Can't find a mask from the mask image." << endl << endl;
    }

    waitKey();
    destroyAllWindows();

    return 0;
}
