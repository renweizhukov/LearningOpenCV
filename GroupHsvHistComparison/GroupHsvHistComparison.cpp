/*
 * GroupHsvHistComparison.cpp
 *
 *  Created on: Sep 13, 2017
 *      Author: renwei
 */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <boost/program_options.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
namespace po = boost::program_options;

void Str2HistComparisonMethod(
    string& strCompMethod,
    vector<int>& compMethods,
    vector<float>& perfectMatchVal)
{
    transform(strCompMethod.begin(), strCompMethod.end(), strCompMethod.begin(), ::tolower);

    compMethods.clear();

    if ((strCompMethod == "correl") || (strCompMethod == "all"))
    {
        compMethods.push_back(CV_COMP_CORREL);
        perfectMatchVal.push_back(1.0);
    }

    if ((strCompMethod == "chisqr") || (strCompMethod == "all"))
    {
        compMethods.push_back(CV_COMP_CHISQR);
        perfectMatchVal.push_back(0.0);
    }

    if ((strCompMethod == "chisqr_alt") || (strCompMethod == "all"))
    {
        compMethods.push_back(CV_COMP_CHISQR_ALT);
        perfectMatchVal.push_back(0.0);
    }

    if ((strCompMethod == "intersect") || (strCompMethod == "all"))
    {
        compMethods.push_back(CV_COMP_INTERSECT);
        perfectMatchVal.push_back(1.0);
    }

    if ((strCompMethod == "bhattacharyya") || (strCompMethod == "hellinger") || (strCompMethod == "all"))
    {
        // CV_COMP_HELLINGER is the same as CV_COMP_BHATTACHARYYA
        compMethods.push_back(CV_COMP_BHATTACHARYYA);
        perfectMatchVal.push_back(0.0);
    }

    if ((strCompMethod == "kl_div") || (strCompMethod == "all"))
    {
        compMethods.push_back(CV_COMP_KL_DIV);
        perfectMatchVal.push_back(0.0);
    }

    return;
}

string HistComparisonMethod2Str(const int histComparisonMethod)
{
    switch (histComparisonMethod)
    {
    case CV_COMP_CORREL:
        return "correl";

    case CV_COMP_CHISQR:
        return "chisqr";

    case CV_COMP_CHISQR_ALT:
        return "chisqr_alt";

    case CV_COMP_INTERSECT:
        return "intersect";

    case CV_COMP_BHATTACHARYYA:
        return "bhattacharyya";

    case CV_COMP_KL_DIV:
        return "kl_dlv";

    default:
        return "invalid";
    }
}

void Str2HsvChannels(
    string& strHsvChannels,
    vector<int>& hsvChannels,
    vector<int>& histSize,
    vector<float>& ranges)
{
    transform(strHsvChannels.begin(), strHsvChannels.end(), strHsvChannels.begin(), ::tolower);

    hsvChannels.clear();
    histSize.clear();
    ranges.clear();

    if (strHsvChannels.find('h') != string::npos)
    {
        hsvChannels.push_back(0);   // Channel 0 is hue.
        histSize.push_back(30);
        ranges.push_back(0);        // Hue range is [0, 179].
        ranges.push_back(180);      // Note that the range is left inclusive and right exclusive.
    }

    if (strHsvChannels.find('s') != string::npos)
    {
        hsvChannels.push_back(1);   // Channel 1 is saturation.
        histSize.push_back(32);
        ranges.push_back(0);        // Saturation range is [0, 255].
        ranges.push_back(256);      // Note that the range is left inclusive and right exclusive.
    }

    if (strHsvChannels.find('v') != string::npos)
    {
        hsvChannels.push_back(2);   // Channel 2 is value.
        histSize.push_back(32);
        ranges.push_back(0);        // Value range is [0, 255].
        ranges.push_back(256);      // Note that the range is left inclusive and right exclusive.
    }
}

// Courtesy of https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv.
string CvType2Str(const int type)
{
    string typeStr("CV_");

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        typeStr += "8U";
        break;

    case CV_8S:
        typeStr += "8S";
        break;

    case CV_16U:
        typeStr += "16U";
        break;

    case CV_16S:
        typeStr += "16S";
        break;

    case CV_32S:
        typeStr += "32S";
        break;

    case CV_32F:
        typeStr += "32F";
        break;

    case CV_64F:
        typeStr += "64F";
        break;

    default:
        typeStr += "User";
        break;
    }

    typeStr += "C";
    typeStr += (chans+'0');

    return typeStr;
}

int GetDirFiles(const string& dir, vector<string>& files)
{
    DIR *dp = nullptr;
    struct dirent *dirp = nullptr;

    dp = opendir(dir.c_str());
    if (dp == nullptr)
    {
        printf("[ERROR]: opendir(%s) opening %s.\n\n", strerror(errno), dir.c_str());
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
            printf("[ERROR]: stat(%s) for %s.\n\n", strerror(errno), dirp->d_name);
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

int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("image1", po::value<string>()->required(), "The first baseline image for the histogram comparison")  // This is a positional option.
        ("image2", po::value<string>()->required(), "The second baseline image for the histogram comparison") // This is also a positional option.
        ("directory,d", po::value<string>()->required(), "The directory of source images for the histogram comparison")
        ("help,h", "Display the help information")
        ("comparison-method,m", po::value<string>(), "The comparison method (correl | chisqr | chisqr_alt | intersect | bhattacharyya | hellinger | kl_div | all). If not specified, default correl.")
        ("hsv-channels,c", po::value<string>(), "The HSV channels used for generating the histogram (h | s | v). If not specified, default hs.");

    po::positional_options_description posOpt;
    posOpt.add("image1", 1);
    posOpt.add("image2", 1);

    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(opt).positional(posOpt).run(), vm);

        if (vm.count("help") > 0)
        {
            printf("Usage: ./SimpleHsvHistComparison image1 image2 -m [comparison-method] -c [HSV-channels]\n\n");
            cout << opt << endl;
            return 0;
        }

        po::notify(vm);
    }
    catch (po::error& e)
    {
        cerr << "[ERROR]: " << e.what() << endl << endl;
        cout << opt << endl;
        return -1;
    }

    string img1;
    string img2;
    string imgDir;
    string strCompMethod;
    string strHsvChannels;

    img1 = vm["image1"].as<string>();
    img2 = vm["image2"].as<string>();
    imgDir = vm["directory"].as<string>();

    if (vm.count("comparison-method") > 0)
    {
        strCompMethod = vm["comparison-method"].as<string>();
    }
    else
    {
        strCompMethod = "correl";
        printf("[INFO]: No distance method is specified and use the default distance method Correlation.\n");
    }

    vector<int> histComparisonMethods;
    vector<float> histCompPerfectMatchVals;
    Str2HistComparisonMethod(strCompMethod, histComparisonMethods, histCompPerfectMatchVals);
    if (histComparisonMethods.empty())
    {
        printf("[ERROR]: Empty histogram comparison method %s.\n\n", strCompMethod.c_str());
        return -1;
    }
    else if (histComparisonMethods.size() > 1)
    {
        printf("[ERROR]: More than one histogram comparison method %s.\n\n", strCompMethod.c_str());
        return -1;
    }

    if (vm.count("hsv-channels") > 0)
    {
        strHsvChannels = vm["hsv-channels"].as<string>();
    }
    else
    {
        strHsvChannels = "hs";
        printf("[INFO]: no hsv channel is specified and use the default channels 0 and 1 (i.e., h and s).\n");
    }

    vector<int> hsvChannels;
    vector<int> histSize;
    vector<float> ranges;
    Str2HsvChannels(strHsvChannels, hsvChannels, histSize, ranges);
    if (hsvChannels.empty())
    {
        printf("[ERROR]: Invalid HSV channels %s should contain h or s or v.\n\n", strHsvChannels.c_str());
        return -1;
    }

    Mat srcImg1 = imread(img1, IMREAD_COLOR);
    if (srcImg1.empty())
    {
        printf("[ERROR]: Cannot load image %s.\n\n", img1.c_str());
        return -1;
    }
    //Mat croppedImg1 = srcImg1(Range(srcImg1.rows/4, srcImg1.rows), Range(0, srcImg1.cols));

    Mat srcImg2 = imread(img2, IMREAD_COLOR);
    if (srcImg2.empty())
    {
        printf("[ERROR]: Cannot load image %s.\n\n", img2.c_str());
        return -1;
    }
    //Mat croppedImg2 = srcImg2(Range(srcImg2.rows/4, srcImg2.rows), Range(0, srcImg2.cols));

    // Get all the image file names in the source directory.
    vector<string> srcFiles;
    int error = GetDirFiles(imgDir, srcFiles);
    if (error != 0)
    {
        return error;
    }

    vector<Mat> srcImgs;
    for (const auto& srcFile: srcFiles)
    {
        Mat srcImg = imread(srcFile);
        //Mat croppedSrcImg = srcImg(Range(srcImg.rows/4, srcImg.rows), Range(0, srcImg.cols));
        srcImgs.push_back(srcImg);
        //srcImgs.push_back(croppedSrcImg);
    }

    // Convert image 1, 2, and all the images in the source directory into HSV.
    Mat hsvImg1;
    Mat hsvImg2;
    cvtColor(srcImg1, hsvImg1, COLOR_BGR2HSV);
    cvtColor(srcImg2, hsvImg2, COLOR_BGR2HSV);
    //cvtColor(croppedImg1, hsvImg1, COLOR_BGR2HSV);
    //cvtColor(croppedImg2, hsvImg2, COLOR_BGR2HSV);

    vector<Mat> hsvSrcImgs;
    for (const auto& srcImg: srcImgs)
    {
        Mat tmpHsvImg;
        cvtColor(srcImg, tmpHsvImg, COLOR_BGR2HSV);
        hsvSrcImgs.push_back(tmpHsvImg);
    }

    // Compute and the normalize the histograms of image 1 and 2, respectively.
    Mat hist1;
    calcHist(vector<Mat>{hsvImg1}, hsvChannels, noArray(), hist1, histSize, ranges);
    normalize(hist1, hist1, 1, 0, NORM_L1);

    Mat hist2;
    calcHist(vector<Mat>{hsvImg2}, hsvChannels, noArray(), hist2, histSize, ranges);
    normalize(hist2, hist2, 1, 0, NORM_L1);

    vector<Mat> hists;
    for (const auto& hsvSrcImg: hsvSrcImgs)
    {
        Mat hist;
        calcHist(vector<Mat>{hsvSrcImg}, hsvChannels, noArray(), hist, histSize, ranges);
        normalize(hist, hist, 1, 0, NORM_L1);

        hists.push_back(hist);
    }

    // Compare the normalized histograms.
    printf("[INFO]: The result for a perfect match of the method %s is %f.\n",
        HistComparisonMethod2Str(histComparisonMethods[0]).c_str(),
        histCompPerfectMatchVals[0]);

    for (size_t histIndex = 0; histIndex < hists.size(); ++histIndex)
    {
        // Compare the source image with image 1.
        double compareResult1 = compareHist(hist1, hists[histIndex], histComparisonMethods[0]);

        printf("[INFO]: The comparison result of source image %s with %s (image 1) = %f.\n",
            srcFiles[histIndex].c_str(), img1.c_str(), compareResult1);

        // Compare the source image with image 2.
        double compareResult2 = compareHist(hist2, hists[histIndex], histComparisonMethods[0]);

        printf("[INFO]: The comparison result of source image %s with %s (image 2) = %f.\n",
            srcFiles[histIndex].c_str(), img2.c_str(), compareResult2);

        double maxCompareResult = max(compareResult1, compareResult2);
        double minCompareResult = min(compareResult1, compareResult2);
        double compareResultRatio = (1 - maxCompareResult )/(1 - minCompareResult);

        printf("[INFO]: The ratio of the distances of comparison results of source image %s to a perfect match = %f.\n\n",
            srcFiles[histIndex].c_str(), compareResultRatio);
    }

    return 0;
}
