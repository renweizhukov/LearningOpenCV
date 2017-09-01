/*
 * main.cpp
 *
 *  Created on: Aug 15, 2017
 *      Author: renwei
 */

#include <iostream>
#include <string>
#include <chrono>
#include <numeric>

#include <boost/program_options.hpp>

#include <opencv2/xfeatures2d.hpp>

#include "Utility.h"
#include "FlannBasedSavableMatcher.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
namespace po = boost::program_options;

typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("command", po::value<string>()->required(), "train | match | help")   // This is a positional option.
        ("image,i", po::value<string>(), "The image file which will be used for SURF matching")
        ("image-dir,d", po::value<string>(), "The directory of images which will be used for training the FLANN-based matcher")
        ("matcher-file,m", po::value<string>(), "The file which will store the FLANN-based matcher. It is an output for training and an input for SURF matching");

    po::positional_options_description posOpt;
    posOpt.add("command", 1);   // Only one command is accepted at one execution.

    po::variables_map vm;
    string cmd;

    try
    {
        po::store(po::command_line_parser(argc, argv).options(opt).positional(posOpt).run(), vm);
        po::notify(vm);
    }
    catch(po::error& e)
    {
        cerr << "[ERROR]: " << e.what() << endl << endl;
        cout << opt << endl;
        return -1;
    }

    // Get the command and convert the command string into lower case.
    cmd = vm["command"].as<string>();
    transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);

    string imgFile;
    string imgDir;
    string matcherFile;

    string matcherFileDir;
    string matcherFilename;

    const int minHessian = 400;
    Ptr<SurfFeatureDetector> detector = SURF::create(minHessian);

    Ptr<FlannBasedSavableMatcher> flannMatcher = FlannBasedSavableMatcher::create();

    if (cmd == "help")
    {
        cout << opt << endl;
    }
    else if (cmd == "train")
    {
        if (vm.count("image-dir") == 0)
        {
            cerr << "[ERROR]: A directory of images is required to be given for training the FLANN-based matcher." << endl << endl;
            return -1;
        }

        if (vm.count("matcher-file") == 0)
        {
            cerr << "[ERROR]: A yml file is required to be given for saving the trained FLANN-based matcher." << endl << endl;
            return -1;
        }

        imgDir = vm["image-dir"].as<string>();
        matcherFile = vm["matcher-file"].as<string>();
        Utility::SeparateDirFromFilename(matcherFile, matcherFileDir, matcherFilename);

        cout << "[INFO]: Loading the images in the given directory." << endl;
        vector<string> imgFilenameList;
        vector<Mat> imgs;

        Utility::LoadAllImagesInDir(imgDir, imgFilenameList, imgs);

        cout << "[INFO]: Detecting the SURF keypoints and computing the descriptors of the images." << endl;

        vector<Mat> allImgDescriptors;
        vector<KeyPoint> oneImgKeypoints;
        Mat oneImgDescriptors;
        for (auto& oneImg: imgs)
        {
            detector->detectAndCompute(oneImg, noArray(), oneImgKeypoints, oneImgDescriptors);
            allImgDescriptors.push_back(oneImgDescriptors);
        }

        cout << "[INFO]: Training the FLANN-based matcher with the SURF descriptors of the images." << endl;

        flannMatcher->add(allImgDescriptors);

        auto tTrainStart = Clock::now();
        flannMatcher->train();
        auto tTrainEnd = Clock::now();
        cout << "[DEBUG]: Trained the FLANN-based matcher in " << chrono::duration_cast<chrono::milliseconds>(tTrainEnd - tTrainStart).count()
            << " ms." << endl;

        cout << "[INFO]: Saving the trained FLANN-based matcher." << endl;

        flannMatcher->setTrainedImgFilenameList(imgFilenameList);
        flannMatcher->setFlannIndexFileDir(matcherFileDir);
        flannMatcher->setFlannIndexFilename(matcherFilename + "_klannindex");

        flannMatcher->save(matcherFile);
    }
    else if (cmd == "match")
    {
        if (vm.count("image") == 0)
        {
            cerr << "[ERROR]: An image is required to be given for doing the SURF matching." << endl << endl;
            return -1;
        }

        if (vm.count("matcher-file") == 0)
        {
            cerr << "[ERROR]: A yml file is required to be given for loading the trained FLANN-based matcher." << endl << endl;
            return -1;
        }

        imgFile = vm["image"].as<string>();
        matcherFile = vm["matcher-file"].as<string>();
        Utility::SeparateDirFromFilename(matcherFile, matcherFileDir, matcherFilename);

        cout << "[INFO]: Loading the given image." << endl;
        Mat img = imread(imgFile);
        if (img.empty())
        {
            cerr << "[ERROR]: Can't load the image " << imgFile << "." << endl << endl;
            return -1;
        }

        cout << "[INFO]: Detecting the SURF keypoints and computing the descriptors of the image." << endl;
        vector<KeyPoint> imgKeypoints;
        Mat imgDescriptors;
        detector->detectAndCompute(img, noArray(), imgKeypoints, imgDescriptors);

        cout << "[INFO]: Loading the trained FLANN-based matcher." << endl;

        // Note that imgFilenameList and flannIndexFilename is saved in the matcherFile and will be loaded automatically in load(),
        // so there is no need to set them here.
        flannMatcher->setFlannIndexFileDir(matcherFileDir);

        auto tFsStart = Clock::now();
        FileStorage fs(matcherFile, FileStorage::READ);
        auto tFsEnd = Clock::now();
        cout << "[DEBUG]: Create the FileStorage for the matcher file in " << chrono::duration_cast<chrono::milliseconds>(tFsEnd - tFsStart).count()
            << " ms." << endl;

        auto tReadStart = Clock::now();
        flannMatcher->read(fs.getFirstTopLevelNode());
        auto tReadEnd = Clock::now();
        cout << "[DEBUG]: Loaded the matcher in " << chrono::duration_cast<chrono::milliseconds>(tReadEnd - tReadStart).count()
            << " ms." << endl;

        vector<string> trainedImgFilenameList;
        trainedImgFilenameList = flannMatcher->getTrainedImgFilenameList();

        cout << "[INFO]: Doing the FLANN-based matching." << endl;

        // Match the descriptor vectors between the target image and all trained images.
        // There are two knnMatch methods: one for object recognition and the other for tracking.
        // Here we are using the one for object recognition.
        vector<vector<DMatch>> allKnnMatches;
        auto tMatchStart = Clock::now();
        flannMatcher->knnMatch(imgDescriptors, allKnnMatches, 2);
        auto tMatchEnd = Clock::now();
        cout << "[DEBUG]: Did the KNN match in " << chrono::duration_cast<chrono::milliseconds>(tMatchEnd - tMatchStart).count()
            << " ms." << endl;

        // Find only "good" matches among the closest matches, i.e., whose distance is much better (<0.8) than
        // the corresponding second closest match.
        vector<DMatch> allGoodMatches;
        vector<int> goodMatchCnts(trainedImgFilenameList.size());
        for (auto& knnMatchPair: allKnnMatches)
        {
            if (knnMatchPair.size() > 1 && knnMatchPair[0].distance < 0.8 * knnMatchPair[1].distance)
            {
                allGoodMatches.push_back(knnMatchPair[0]);
                goodMatchCnts[knnMatchPair[0].imgIdx]++;
            }
        }

        // Find the best matched source image (i.e., with the most "good matches").
        auto bestMatchImageIt = max_element(goodMatchCnts.begin(), goodMatchCnts.end());
        int bestMatchImageIndex = distance(goodMatchCnts.begin(), bestMatchImageIt);

        double bestMatchedPercentage = 100.0 * (*bestMatchImageIt) / imgDescriptors.rows;

        cout << "[INFO]: The best matched trained image is " << trainedImgFilenameList[bestMatchImageIndex]
            << " with the matched percentage " << bestMatchedPercentage << "%." << endl;

        double overallMatchedPercentage = 100.0 * accumulate(goodMatchCnts.begin(), goodMatchCnts.end(), 0) / imgDescriptors.rows;
        cout << "[INFO]: The overall matched percentage is " << overallMatchedPercentage << "%." << endl;
    }
    else
    {
        cerr << "[ERROR]: " << "Unknown command " << cmd << endl << endl;
        cout << opt << endl;
        return -1;
    }

    return 0;
}
