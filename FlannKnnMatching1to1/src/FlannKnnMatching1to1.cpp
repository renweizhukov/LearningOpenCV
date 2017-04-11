/*
 * FlannKnnMatching1to1.cpp
 *
 *  Created on: Mar 31, 2017
 *      Author: Wei Ren
 */

#include <cstdio>
#include <vector>
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

typedef std::chrono::high_resolution_clock Clock;

// Forward definition of the function readme().
void readme();

/*
 * @function main
 * @brief Main function
 */
int main(int argc, char** argv)
{
    if( argc != 3 )
    {
        readme();
        return -1;
    }

    // 0. Load the two images.
    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);

    if (img1.empty() || img2.empty())
    {
        printf("Error reading images!!\n");
        return -1;
    }

    auto tStart = Clock::now();

    // 1. Detect the keypoints using the SURF detector and compute the descriptors which
    // somehow describe the keypoints and can be used for comparing two sets of keypoints.
    const int minHessian = 400;
    Ptr<SURF> surfDetector = SURF::create(minHessian);

    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat descriptors1;
    Mat descriptors2;

    surfDetector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    surfDetector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    auto tDetectAndComputeEnd = Clock::now();

    printf("======================================================================\n");
    printf("The time for the SURF keypoint detection and the descriptor computation: %ld milliseconds\n",
            chrono::duration_cast<chrono::milliseconds>(tDetectAndComputeEnd - tStart).count());

    // 2. Match the two descriptors using the FLANN matcher.
    FlannBasedMatcher flannMatcher;
    vector<vector<DMatch>> knnMatches;

    // There are two match methods: one for object recognition and the other for tracking.
    // Here we are using the one for tracking.
    flannMatcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    auto tMatchEnd = Clock::now();

    printf("======================================================================\n");
    printf("The time for the FLANN matching: %ld milliseconds\n",
            chrono::duration_cast<chrono::milliseconds>(tMatchEnd - tDetectAndComputeEnd).count());

    // 3. Find only "good" matches among the closest matches, i.e., whose distance is much better (<0.75) than
    // the corresponding second closest match.
    vector<DMatch> goodMatches;
    for (auto& knnMatch: knnMatches)
    {
        if (knnMatch.size() > 1 && knnMatch[0].distance < 0.75 * knnMatch[1].distance)
        {
            goodMatches.push_back(knnMatch[0]);
        }
    }

    // Draw only "good" matches.
    Mat imgMatches;
    drawMatches(img1, keypoints1, img2, keypoints2,
            goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    namedWindow("Good matches", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);

    // Display the two images and connect the "good"-matched keypoints.
    imshow("Good matches", imgMatches);

    // Display the percentage of "good" matches.
    printf("======================================================================\n");
    printf("Good Match percentage = %f%%\n", 100.0 * goodMatches.size() / keypoints1.size());

    // Display detected matches in the console.
    printf("======================================================================\n");
    for (int goodMatchIndex = 0; goodMatchIndex < static_cast<int>(goodMatches.size()); goodMatchIndex++)
    {
        printf("Good Match [%d]: keypoint 1: %d <--> keypoint 2: %d with distance = %f\n",
                goodMatchIndex, goodMatches[goodMatchIndex].queryIdx, goodMatches[goodMatchIndex].trainIdx,
                goodMatches[goodMatchIndex].distance);
    }

    waitKey(0);

    return 0;
}

/*
 * @function readme()
 */
void readme()
{
    printf(" Usage: ./FlannKnnMatching1to1 <img1> <img2>\n");
}
