/*
 * main.cpp
 *
 *  Created on: Aug 15, 2017
 *      Author: renwei
 */

#include <iostream>
#include <string>
#include <map>
#include <chrono>
#include <numeric>

#include <boost/program_options.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "Utility.h"
#include "FlannBasedSavableMatcher.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
namespace po = boost::program_options;

typedef std::chrono::high_resolution_clock Clock;

struct FnnMatchResult
{
    string expectedLabel;
    string evaluatedLabel;
    string candidateLabel;
    float overallGoodMatchPercent;
    int overallGoodMatchCnt;
    float pairwiseGoodMatchPercent;

    FnnMatchResult()
    {
    }

    // Write serialization for this class
    void write(cv::FileStorage& fs) const
    {
        fs << "{" << "expectedLabel" << expectedLabel;
        fs << "evaluatedLabel" << evaluatedLabel;
        fs << "candidateLabel" << candidateLabel;
        fs << "overallGoodMatchPercent" << overallGoodMatchPercent;
        fs << "overallGoodMatchCnt" << overallGoodMatchCnt;
        fs << "pairwiseGoodMatchPercent" << pairwiseGoodMatchPercent << "}";
    }

    // Read de-serialization for this class
    void read(const cv::FileNode& node)
    {
        expectedLabel = (std::string)(node["expectedLabel"]);
        evaluatedLabel = (std::string)(node["evaluatedLabel"]);
        candidateLabel = (std::string)(node["candidateLabel"]);
        overallGoodMatchPercent = (float)(node["overallGoodMatchPercent"]);
        overallGoodMatchCnt = (int)(node["overallGoodMatchCnt"]);
        pairwiseGoodMatchPercent = (float)(node["pairwiseGoodMatchPercent"]);
    }
};

static void write(
    FileStorage& fs,
    const string&,
    const FnnMatchResult& result)
{
    result.write(fs);
}

static void read(
    const FileNode& node,
    FnnMatchResult& result,
    const FnnMatchResult& defaultValue = FnnMatchResult())
{
    if (node.empty())
    {
        result = defaultValue;
    }
    else
    {
        result.read(node);
    }
}

void InitFlannBasedMatcher(
    Ptr<FlannBasedSavableMatcher>& flannMatcher,
    const string& matcherFileDir,
    const string& matcherFile)
{
    flannMatcher->setFlannIndexFileDir(matcherFileDir);

    auto tStart = Clock::now();
    FileStorage fs(matcherFile, FileStorage::READ);
    auto tEnd = Clock::now();
    cout << "[DEBUG]: InitFlannBasedMatcher::FileStorage in " << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " ms." << endl;

    flannMatcher->read(fs.getFirstTopLevelNode());
}

void FlannBasedKnnMatch(
    const Mat& imgDescriptors,
    const Ptr<FlannBasedSavableMatcher>& flannMatcher,
    const vector<pair<string, string> >& matcherTrainedImg2LabelList,
    const string& imgMapKey,
    FnnMatchResult& result)
{
    const float overallGoodMatchPercentUpperThreshold = 7.5;
    const float overallGoodMatchPercentLowerThreshold = overallGoodMatchPercentUpperThreshold/3;
    const int overallGoodMatchCntThreshold = 10;

    string evaluatedLabel("unknown");
    string candidateLabel("unknown");
    float overallGoodMatchPercent = 0.0;
    int overallGoodMatchCnt = 0;
    float pairwiseGoodMatchPercent = 0.0;

    vector<vector<DMatch>> overallKnnMatches;
    flannMatcher->knnMatch(imgDescriptors, overallKnnMatches, 2);

    vector<int> overallMatchCnts(matcherTrainedImg2LabelList.size());
    vector<int> overallGoodMatchCnts(matcherTrainedImg2LabelList.size());
    for (const auto& knnMatchPair: overallKnnMatches)
    {
        if (knnMatchPair.size() > 1 && knnMatchPair[0].distance < 0.75 * knnMatchPair[1].distance)
        {
            overallGoodMatchCnts[knnMatchPair[0].imgIdx]++;
        }

        if (!knnMatchPair.empty())
        {
            overallMatchCnts[knnMatchPair[0].imgIdx]++;
        }
    }

    auto bestOverallMatchImageIt = max_element(overallGoodMatchCnts.begin(), overallGoodMatchCnts.end());
    int bestOverallMatchImageIndex = distance(overallGoodMatchCnts.begin(), bestOverallMatchImageIt);

    candidateLabel = matcherTrainedImg2LabelList[bestOverallMatchImageIndex].second;
    overallGoodMatchCnt = *bestOverallMatchImageIt;
    overallGoodMatchPercent = 100.0*overallGoodMatchCnt/(imgDescriptors.rows);

    if (overallGoodMatchCnt >= overallGoodMatchCntThreshold)
    {
        if (overallGoodMatchPercent >= overallGoodMatchPercentUpperThreshold)
        {
            evaluatedLabel = candidateLabel;
            cout << "[INFO]: The maximum overall match percentage " << overallGoodMatchPercent << "% of " << imgMapKey
                << " is above the upper threshold " << overallGoodMatchPercentUpperThreshold << "%, so evaluate the class as "
                << evaluatedLabel << "." << endl;
        }
        else if (overallGoodMatchPercent >= overallGoodMatchPercentLowerThreshold)
        {
            // The overall good match percentage is marginally low, i.e., between the lower threshold and the upper threshold.
            // In this case, we do a pairwise knnMatch between the query image and the best overall match to see if the pairwise
            // good match percentage is greater than or equal to the upper threshold. Since the descriptors of images other than
            // the best overall match are excluded in the pairwise knnMatch, it usually yields a greater good match percentage.

            vector<Mat> allTrainedDescriptors = flannMatcher->getTrainDescriptors();
            Mat candidateImgDescriptors = allTrainedDescriptors[bestOverallMatchImageIndex];

            vector<vector<DMatch> > knnMatches;
            flannMatcher->knnMatch(imgDescriptors, candidateImgDescriptors, knnMatches, 2);
            int pairwiseGoodMatchCnt = 0;
            for (const auto& knnMatchPair : knnMatches)
            {
                if (knnMatchPair.size() > 1 && knnMatchPair[0].distance < 0.75 * knnMatchPair[1].distance)
                {
                    ++pairwiseGoodMatchCnt;
                }
            }

            pairwiseGoodMatchPercent = 100.0*pairwiseGoodMatchCnt/(imgDescriptors.rows);
            if (pairwiseGoodMatchPercent >= overallGoodMatchPercentUpperThreshold)
            {
                evaluatedLabel = candidateLabel;
                cout << "[INFO]: Although the maximum overall match percentage " << overallGoodMatchPercent << "% of " << imgMapKey
                << " is below the upper threshold " << overallGoodMatchPercentUpperThreshold << "%, the 1-to-1 match percentage "
                << pairwiseGoodMatchPercent << " is above the threshold " << overallGoodMatchPercentUpperThreshold << ", so evaluate the class as"
                << evaluatedLabel << "." << endl;
            }
            else
            {
                cout << "[INFO]: Both the maximum overall match percentage " << overallGoodMatchPercent << "% and the 1-to-1 match percentage "
                    << pairwiseGoodMatchPercent << "% of " << imgMapKey << " is below the upper threshold "
                    << overallGoodMatchPercentUpperThreshold << "%, so evaluate the class as unknown." << endl;
            }
        }
        else
        {
            cout << "[INFO]: The maximum overall match percentage " << overallGoodMatchPercent << "% of " << imgMapKey
                << " is below the lower threshold " << overallGoodMatchPercentLowerThreshold << "%, so evaluate the class as unknown." << endl;
        }
    }
    else
    {
        // The number of overall good matches is too small, and it implies that the query image has few descriptors,
        // possibly a blank page.
        cout << "[INFO]: The maximum overall match count " << overallGoodMatchCnt << "% of " << imgMapKey
            << " is below the threshold " << overallGoodMatchCntThreshold << ", so evaluate the class as unknown." << endl;
    }

    result.evaluatedLabel = evaluatedLabel;
    result.candidateLabel = candidateLabel;
    result.overallGoodMatchPercent = overallGoodMatchPercent;
    result.overallGoodMatchCnt = overallGoodMatchCnt;
    result.pairwiseGoodMatchPercent = pairwiseGoodMatchPercent;
}

void WriteResultsToFile(
    const map<string, FnnMatchResult>& img2ResultMap,
    const string& resultFile)
{
    cout << "===============================================================================================" << endl;
    cout << "[INFO]: Complete results of " << img2ResultMap.size() << " images:" << endl;
    cout << "===============================================================================================" << endl;

    map<string, FnnMatchResult> img2ResultErrors;
    for (auto& img2Result : img2ResultMap)
    {
        if (img2Result.second.evaluatedLabel != img2Result.second.expectedLabel)
        {
            img2ResultErrors.insert(img2Result);
        }

        cout << "[INFO]: " << img2Result.first << ": expected label = " << img2Result.second.expectedLabel
            << ", evaluated label = " << img2Result.second.evaluatedLabel << ", candidate label = "
            << img2Result.second.candidateLabel << " with matchPercent = " << img2Result.second.overallGoodMatchPercent
            << "% and matchCnt = " << img2Result.second.overallGoodMatchCnt << "." << endl;
    }

    cout << "===============================================================================================" << endl;
    cout << "[INFO]: Error rate = " << 100.0*img2ResultErrors.size()/img2ResultMap.size() << "%." << endl;
    cout << "[INFO]: " << img2ResultErrors.size() << " error images:" << endl;
    cout << "===============================================================================================" << endl;

    for (const auto& img2Result : img2ResultErrors)
    {
        cout << "[INFO]: " << img2Result.first << ": expected label = " << img2Result.second.expectedLabel
                << ", evaluated class = " << img2Result.second.evaluatedLabel << "." << endl;
    }

    FileStorage fsResult(resultFile, FileStorage::WRITE);

    // Write the detailed classifier result of the image to the result file.
    for (const auto& img2Result : img2ResultMap)
    {
        // For OpenCV FileStorage, key names may only contain alphanumeric characters [a-zA-Z0-9],
        // '-', '_' and ' '. Unfortunately key names may not contain '.'. Also key names must
        // start with a letter or '_'. Since the image filename may start with a non-letter,
        // e.g., a digit, we have to put it after those prefixes.
        size_t dotPos = img2Result.first.find_last_of('.');
        string img2ResultFsKey = "classifier_result_" + img2Result.first.substr(0, dotPos);

        fsResult << img2ResultFsKey << img2Result.second;
    }

    fsResult.release();
}

int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("command", po::value<string>()->required(), "train | match | help")   // This is a positional option.
        ("expected-label,l", po::value<string>(), "The expected label of the image file for SURF matching")
        ("image,i", po::value<string>(), "The image file which will be used for SURF matching")
        ("image-dir,d", po::value<string>(), "The directory of images which will be used for training the FLANN-based matcher or doing the SURF matching")
        ("matcher-file,m", po::value<string>(), "The file which will store the FLANN-based matcher. It is an output for training and an input for SURF matching")
        ("result,r", po::value<string>(), "The output file which will store the matching results");

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

    string expectedLabel;
    string imgFile;
    string imgDir;
    string matcherFile;
    string resultFile;

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

        cout << "[INFO]: Loading the images, detecting the SURF keypoints and computing the descriptors." << endl;
        vector<pair<string, string> > label2Imgs;
        Utility::GetImagesWithLabels(imgDir, label2Imgs);

        vector<pair<string, string> > trainedImgFilename2LabelList;
        vector<Mat> allImgDescriptors;
        for (const auto& label2Img : label2Imgs)
        {
            string imgLabel = label2Img.first;
            string imgFilename = label2Img.second;
            string imgFullFilename = imgDir + "/" + imgLabel + "/" + imgFilename;

            trainedImgFilename2LabelList.push_back(make_pair(imgFilename, imgLabel));

            Mat img = imread(imgFullFilename);

            vector<KeyPoint> oneImgKeypoints;
            Mat oneImgDescriptors;
            detector->detectAndCompute(img, noArray(), oneImgKeypoints, oneImgDescriptors);
            allImgDescriptors.push_back(oneImgDescriptors);
        }

        cout << "[INFO]: Training the FLANN-based matcher with the SURF descriptors of the images." << endl;

        flannMatcher->add(allImgDescriptors);

        auto tTrainStart = Clock::now();
        flannMatcher->train();
        auto tTrainEnd = Clock::now();
        cout << "[INFO]: Trained the FLANN-based matcher in " << chrono::duration_cast<chrono::milliseconds>(tTrainEnd - tTrainStart).count()
            << " ms." << endl;

        cout << "[INFO]: Saving the trained FLANN-based matcher." << endl;

        flannMatcher->setTrainedImgFilename2LabelList(trainedImgFilename2LabelList);
        flannMatcher->setFlannIndexFileDir(matcherFileDir);
        flannMatcher->setFlannIndexFilename(matcherFilename + "_klannindex");

        flannMatcher->save(matcherFile);
    }
    else if (cmd == "match")
    {
        if ((vm.count("image") > 0) && (vm.count("expected-label") == 0))
        {
            cerr << "[ERROR]: The expected label of the test image is required to be compared with the evaluated label." << endl << endl;
            return -1;
        }

        if ((vm.count("image") == 0) && (vm.count("image-dir") == 0))
        {
            cerr << "[ERROR]: Either an image or an image base path is required to be given for testing the SURF matching." << endl << endl;
            return -1;
        }

        if (vm.count("matcher-file") == 0)
        {
            cerr << "[ERROR]: A matcher file is required to be given for loading the trained FLANN-based matcher." << endl << endl;
            return -1;
        }

        if (vm.count("result") == 0)
        {
            cerr << "[ERROR]: A file is required to be given for recording the testing result." << endl << endl;
            return -1;
        }

        matcherFile = vm["matcher-file"].as<string>();
        Utility::SeparateDirFromFilename(matcherFile, matcherFileDir, matcherFilename);

        resultFile = vm["result"].as<string>();

        map<string, string> img2FullFilenameMap;
        map<string, FnnMatchResult> img2ResultMap;
        string imgMapKey;
        if (vm.count("image") > 0)
        {
            expectedLabel = vm["expected-label"].as<string>();
            imgFile = vm["image"].as<string>();

            string imgFilename;
            Utility::SeparateDirFromFilename(imgFile, imgDir, imgFilename);
            imgMapKey = expectedLabel + "_" + imgFilename;

            img2FullFilenameMap.insert(make_pair(imgMapKey, imgFile));

            FnnMatchResult res;
            res.expectedLabel = expectedLabel;
            res.evaluatedLabel = "unknown";
            res.candidateLabel = "unknown";
            res.overallGoodMatchPercent = 0.0;
            res.overallGoodMatchCnt = 0;
            res.pairwiseGoodMatchPercent = 0.0;
            img2ResultMap.insert(make_pair(imgMapKey, res));
        }
        else
        {
            // i.e., vm.count("image-dir") > 0
            imgDir = vm["image-dir"].as<string>();

            vector<pair<string, string> > label2Imgs;
            Utility::GetImagesWithLabels(imgDir, label2Imgs);

            for (const auto& label2Img : label2Imgs)
            {
                string imgLabel = label2Img.first;
                string imgFilename = label2Img.second;
                string imgFullFilename = imgDir + "/" + imgLabel + "/" + imgFilename;

                imgMapKey = imgLabel + "_" + imgFilename;

                img2FullFilenameMap.insert(make_pair(imgMapKey, imgFullFilename));

                FnnMatchResult res;
                res.expectedLabel = imgLabel;
                res.evaluatedLabel = "unknown";
                res.overallGoodMatchPercent = 0.0;
                res.overallGoodMatchCnt = 0;
                res.pairwiseGoodMatchPercent = 0.0;
                img2ResultMap.insert(make_pair(imgMapKey, res));
            }
        }

        cout << "[INFO]: Loading the trained FLANN-based matcher." << endl;

        // Note that imgFilenameList and flannIndexFilename is saved in the matcherFile and will be loaded automatically in load(),
        // so there is no need to set them here.
        auto tLoadStart = Clock::now();
        InitFlannBasedMatcher(flannMatcher, matcherFileDir, matcherFile);
        auto tLoadEnd = Clock::now();

        vector<pair<string, string> > matcherTrainedImg2LabelList;
        matcherTrainedImg2LabelList = flannMatcher->getTrainedImgFilename2LabelList();

        cout << "[INFO]: Loaded the FLANN-based matcher for " << matcherTrainedImg2LabelList.size() << " labels in "
            << chrono::duration_cast<chrono::milliseconds>(tLoadEnd - tLoadStart).count() << " ms." << endl;

        auto tMatchStart = Clock::now();
        for (auto& img2Result : img2ResultMap)
        {
            string imgFullFilename = img2FullFilenameMap[img2Result.first];

            cout << "[INFO]: Loading the image " << imgFullFilename << "." << endl;
            Mat img = imread(imgFullFilename);
            if (img.empty())
            {
                cerr << "[ERROR]: Can't load the image " << imgFullFilename << "." << endl << endl;
                return -1;
            }

            cout << "[INFO]: Detecting the SURF keypoints and computing the descriptors of the image "
                << imgFullFilename << "." << endl;
            vector<KeyPoint> imgKeypoints;
            Mat imgDescriptors;
            detector->detectAndCompute(img, noArray(), imgKeypoints, imgDescriptors);

            cout << "[INFO]: Doing the FLANN-based knnMatching for the image " << imgFullFilename << "." << endl;
            FlannBasedKnnMatch(
                imgDescriptors,
                flannMatcher,
                matcherTrainedImg2LabelList,
                img2Result.first,
                img2Result.second);
        }
        auto tMatchEnd = Clock::now();
        cout << "[INFO]: Did the FLANN-based knnMatching for " << img2FullFilenameMap.size() << " images in "
            << chrono::duration_cast<chrono::milliseconds>(tMatchEnd - tMatchStart).count() << " ms." << endl;

        WriteResultsToFile(img2ResultMap, resultFile);
    }
    else
    {
        cerr << "[ERROR]: " << "Unknown command " << cmd << endl << endl;
        cout << opt << endl;
        return -1;
    }

    return 0;
}
