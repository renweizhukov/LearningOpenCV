/*
 * FlannKnnMatching1toN.cpp
 *
 *  Created on: Mar 31, 2017
 *      Author: Wei Ren
 */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#include <iterator>
#include <algorithm>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

typedef std::chrono::high_resolution_clock Clock;

const double presetMaxGoodDist = 0.02;

/*
 * @function Readme()
 */
void Readme()
{
    printf(" Usage: ./FlannKnnMatching1toN [surf/orb] [slow/fast] <source_image_directory> <target_image>\n");
}

/*
 * @function GetDirFiles()
 */
int GetDirFiles(const string& dir, vector<string>& files)
{
    DIR *dp = nullptr;
    struct dirent *dirp = nullptr;

    dp = opendir(dir.c_str());
    if (dp == nullptr)
    {
        printf("Error: opendir(%s) opening %s.\n", strerror(errno), dir.c_str());
        return errno;
    }

    struct stat info;

    files.clear();
    string completedDir(dir);
    if (completedDir.back() != '/')
    {
        completedDir.push_back('/');
    }

    while ((dirp = readdir(dp)) != nullptr)
    {
        string fileFullPath = completedDir + string(dirp->d_name);

        if (stat(fileFullPath.c_str(), &info) != 0)
        {
            printf("Error: stat(%s) for %s.\n", strerror(errno), dirp->d_name);
            continue;
        }

        if (S_ISREG(info.st_mode))
        {
            files.push_back(fileFullPath);
        }
    }

    closedir(dp);
    return 0;
}

/*
 * @function FindBestMatchedImageSlow()
 */
int FindBestMatchedImageSlow(
    const vector<string>& srcFiles,
    const Ptr<FlannBasedMatcher>& matcher,
    const vector<Mat>& allSrcDescriptors,
    const Mat& targetDescriptors,
    vector<vector<DMatch>>& allKnnMatches,
    vector<DMatch>& allGoodMatches,
    double& matchedPercentage)
{
    vector<int> matchCnts(allSrcDescriptors.size());
    vector<int> goodMatchCnts(allSrcDescriptors.size());

    auto tMatchStart = Clock::now();

    // Match the descriptor vectors between the target image and each source image.
    for (int srcIndex = 0; srcIndex < static_cast<int>(allSrcDescriptors.size()); srcIndex++)
    {
        // There are two knnMatch methods: one for object recognition and the other for tracking.
        // Here we are using the one for tracking.
        vector<vector<DMatch>> oneSrcKnnMatches;
        matcher->knnMatch(targetDescriptors, allSrcDescriptors[srcIndex], oneSrcKnnMatches, 2);

        // Set the imgIdx of oneMatches. Also find only "good" matches among the closest matches,
        // i.e., whose distance is much better (<0.75) than the corresponding second closest match.
        for (auto& knnMatchPair: oneSrcKnnMatches)
        {
            for (auto& knnMatch: knnMatchPair)
            {
                knnMatch.imgIdx = srcIndex;
            }

            if (knnMatchPair.size() > 1 && knnMatchPair[0].distance < 0.75 * knnMatchPair[1].distance)
            {
                allGoodMatches.push_back(knnMatchPair[0]);
                goodMatchCnts[knnMatchPair[0].imgIdx]++;
            }

            if (!knnMatchPair.empty())
            {
                matchCnts[knnMatchPair[0].imgIdx]++;
            }
        }
    }

    auto tMatchEnd = Clock::now();

    printf("======================================================================\n");
    printf("The time for matching the descriptors of %lu images: %ld milliseconds\n",
            allSrcDescriptors.size(), chrono::duration_cast<chrono::milliseconds>(tMatchEnd - tMatchStart).count());

    // Find the best matched source image (i.e., with the most "good matches").
    auto bestMatchImageIt = max_element(goodMatchCnts.begin(), goodMatchCnts.end());
    int bestMatchImageIndex = distance(goodMatchCnts.begin(), bestMatchImageIt);

    matchedPercentage = 100.0 * (*bestMatchImageIt) / targetDescriptors.rows;

    printf("======================================================================\n");
    printf("The index of the best matched source image is %d with the matched percentage %f%%\n",
            bestMatchImageIndex, matchedPercentage);

    printf("======================================================================\n");
    printf("The counts of matches and good matches for each source image: \n");
    for (size_t srcImgIndex = 0; srcImgIndex < allSrcDescriptors.size(); srcImgIndex++)
    {
        printf("\tsrc image %s: match# = %d, good match# = %d\n",
                srcFiles[srcImgIndex].c_str(), matchCnts[srcImgIndex], goodMatchCnts[srcImgIndex]);
    }

    return bestMatchImageIndex;
}

/*
 * @function FindBestMatchedImageFast()
 */
int FindBestMatchedImageFast(
    const vector<string>& srcFiles,
    const Ptr<FlannBasedMatcher>& matcher,
    const vector<Mat>& allSrcDescriptors,
    const Mat& targetDescriptors,
    vector<vector<DMatch>>& allKnnMatches,
    vector<DMatch>& allGoodMatches,
    double& matchedPercentage)
{
    // Add the descriptors of the source images as the train descriptors.
    matcher->add(allSrcDescriptors);

    // Train the matcher. Note that this train step may not be necessary, since at the beginning of
    // each match() function it will train the matcher if the train step is not done.
    // For details, please see:
    // http://stackoverflow.com/questions/31744451/descriptormatcher-opencv-train
    matcher->train();

    auto tMatchStart = Clock::now();

    // Match the descriptor vectors between the target image and all source images.
    // There are two knnMatch methods: one for object recognition and the other for tracking.
    // Here we are using the one for object recognition.
    matcher->knnMatch(targetDescriptors, allKnnMatches, 2);

    auto tMatchEnd = Clock::now();

    printf("======================================================================\n");
    printf("The time for matching the descriptors of %lu images: %ld milliseconds\n",
            allSrcDescriptors.size(), chrono::duration_cast<chrono::milliseconds>(tMatchEnd - tMatchStart).count());

    // Find only "good" matches among the closest matches, i.e., whose distance is much better (<0.75) than
    // the corresponding second closest match.
    vector<int> matchCnts(allSrcDescriptors.size());
    vector<int> goodMatchCnts(allSrcDescriptors.size());
    for (auto& knnMatchPair: allKnnMatches)
    {
        if (knnMatchPair.size() > 1 && knnMatchPair[0].distance < 0.75 * knnMatchPair[1].distance)
        {
            allGoodMatches.push_back(knnMatchPair[0]);
            goodMatchCnts[knnMatchPair[0].imgIdx]++;
        }

        if (!knnMatchPair.empty())
        {
            matchCnts[knnMatchPair[0].imgIdx]++;
        }
    }

    // Find the best matched source image (i.e., with the most "good matches").
    auto bestMatchImageIt = max_element(goodMatchCnts.begin(), goodMatchCnts.end());
    int bestMatchImageIndex = distance(goodMatchCnts.begin(), bestMatchImageIt);

    matchedPercentage = 100.0 * (*bestMatchImageIt) / targetDescriptors.rows;

    printf("======================================================================\n");
    printf("The index of the best matched source image is %d with the matched percentage %f%%\n",
            bestMatchImageIndex, matchedPercentage);

    printf("======================================================================\n");
    printf("The counts of matches and good matches for each source image: \n");
    for (size_t srcImgIndex = 0; srcImgIndex < allSrcDescriptors.size(); srcImgIndex++)
    {
        printf("\tsrc image %s: match# = %d, good match# = %d\n",
                srcFiles[srcImgIndex].c_str(), matchCnts[srcImgIndex], goodMatchCnts[srcImgIndex]);
    }

    return bestMatchImageIndex;
}

/*
 * @function main
 * @brief Main function
 */
int main(int argc, char** argv)
{
    if(argc != 5)
    {
        Readme();
        return -1;
    }

    // 0. Load the source images as well as the target image.

    // Get all the image file names in the source directory.
    string srcDir(argv[3]);
    vector<string> srcFiles;

    int error = GetDirFiles(srcDir, srcFiles);
    if (error != 0)
    {
        return error;
    }

    vector<Mat> srcImages;
    for (auto& srcFile: srcFiles)
    {
        srcImages.push_back(imread(srcFile));
    }

    Mat targetImage = imread(argv[4]);

    auto tStart = Clock::now();

    // 1. Detect the keypoints using either the SURF detector or the ORB detector, and
    // compute the descriptors which somehow describe the keypoints and can be used
    // for comparing two sets of keypoints.
    string detectorMethod(argv[1]);
    // Convert detectorMethod into a lower-case string.
    transform(detectorMethod.begin(), detectorMethod.end(), detectorMethod.begin(), ::tolower);

    Ptr<Feature2D> detector;
    Ptr<FlannBasedMatcher> matcher;
    if (detectorMethod == "surf")
    {
        const int minHessian = 400;
        detector = SURF::create(minHessian);
        matcher = makePtr<FlannBasedMatcher>();
    }
    else if (detectorMethod == "orb")
    {
        detector = ORB::create();
        matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(6, 12, 1), makePtr<flann::SearchParams>(50));
    }
    else
    {
        printf("Error: unsupported detector method %s. should be either \"surf\" or \"orb\".", detectorMethod.c_str());
        return -1;
    }

    vector<vector<KeyPoint>> allSrcKeypoints;
    vector<Mat> allSrcDescriptors;
    vector<KeyPoint> oneSrcKeypoints;
    Mat oneSrcDescriptors;

    for (auto& srcImage: srcImages)
    {
        detector->detectAndCompute(srcImage, noArray(), oneSrcKeypoints, oneSrcDescriptors);
        allSrcKeypoints.push_back(oneSrcKeypoints);
        allSrcDescriptors.push_back(oneSrcDescriptors);
    }

    vector<KeyPoint> targetKeypoints;
    Mat targetDescriptors;

    detector->detectAndCompute(targetImage, noArray(), targetKeypoints, targetDescriptors);

    auto tDetectAndComputeEnd = Clock::now();

    printf("======================================================================\n");
    printf("The time for the SURF keypoint detection and the descriptor computation of %lu images: %ld milliseconds\n",
            srcImages.size() + 1, chrono::duration_cast<chrono::milliseconds>(tDetectAndComputeEnd - tStart).count());


    vector<vector<DMatch>> allKnnMatches;
    vector<DMatch> allGoodMatches;
    int matchedImgIndex = -1;
    double matchedPercentage = 0.0;

    string matchOption(argv[2]);
    if (matchOption == "slow")
    {
        matchedImgIndex = FindBestMatchedImageSlow(
                srcFiles,
                matcher,
                allSrcDescriptors,
                targetDescriptors,
                allKnnMatches,
                allGoodMatches,
                matchedPercentage);
    }
    else if (matchOption == "fast")
    {
        matchedImgIndex = FindBestMatchedImageFast(
                srcFiles,
                matcher,
                allSrcDescriptors,
                targetDescriptors,
                allKnnMatches,
                allGoodMatches,
                matchedPercentage);
    }
    else
    {
        printf("Error: unsupported match option %s. should be either \"slow\" or \"fast\".", matchOption.c_str());
        return -1;
    }

    auto tMatchEnd = Clock::now();
    printf("======================================================================\n");
    printf("The time for the FLANN matching: %ld milliseconds\n",
            chrono::duration_cast<chrono::milliseconds>(tMatchEnd - tDetectAndComputeEnd).count());

    vector<DMatch> goodMatches;  // "Good" matches between the target image and the best matched image.
    for (auto& goodMatch: allGoodMatches)
    {
        if (goodMatch.imgIdx == matchedImgIndex)
        {
            goodMatches.push_back(goodMatch);
        }
    }

    // Draw only "good" matches.
    Mat imgMatches;
    drawMatches(targetImage, targetKeypoints, srcImages[matchedImgIndex], allSrcKeypoints[matchedImgIndex],
            goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    namedWindow("Good matches", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);

    // Display the two images and connect the "good"-matched keypoints.
    imshow("Good matches", imgMatches);

    // Display the percentage of "good" matches.
    printf("======================================================================\n");
    printf("Good Match percentage = %f%%\n", matchedPercentage);

    // Display detected matches in the console.
    printf("======================================================================\n");
    for (int goodMatchIndex = 0; goodMatchIndex < static_cast<int>(goodMatches.size()); goodMatchIndex++)
    {
        printf("Good Match [%d]: target keypoint: %d <--> source %d keypoint: %d with distance = %f\n",
                goodMatchIndex, goodMatches[goodMatchIndex].queryIdx, matchedImgIndex, goodMatches[goodMatchIndex].trainIdx,
                goodMatches[goodMatchIndex].distance);
    }

    waitKey(0);

    return 0;
}
