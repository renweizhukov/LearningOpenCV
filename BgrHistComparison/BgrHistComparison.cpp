/*
 * BgrHistComparison.cpp
 *
 *  Created on: Jul 26, 2017
 *      Author: renwei
 */

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

// CV_COMP_EMD is not defined in OpenCV, so we define a macro as a workaround.
#define CV_COMP_EMD (CV_COMP_KL_DIV + 1)

void Str2BgrChannels(
    const string& StrBgrChannel,
    vector<int>& bgrChannels,
    vector<int>& histSize,
    vector<float>& ranges)
{
    bgrChannels.clear();
    histSize.clear();
    ranges.clear();

    if (StrBgrChannel.find('b') != string::npos)
    {
        bgrChannels.push_back(0);   // Channel 0 is blue.
    }

    if (StrBgrChannel.find('g') != string::npos)
    {
        bgrChannels.push_back(1);   // Channel 1 is green.
    }

    if (StrBgrChannel.find('r') != string::npos)
    {
        bgrChannels.push_back(2);   // Channel 2 is red.
    }

    // If more than one BGR channel is specified, we will compute and compare the histograms per each channel.
    histSize.push_back(32);
    ranges.push_back(0);        // range for each BGR channel is always [0, 255].
    ranges.push_back(256);      // Note that the range is left inclusive and right exclusive.
}

string BgrChannel2Str(const int& channel)
{
    switch (channel)
    {
    case 0:
        return "blue";

    case 1:
        return "green";

    case 2:
        return "red";

    default:
        return "invalid";
    }
}

void Str2HistComparisonMethod(
    const string& strCompMethod,
    vector<int>& compMethods,
    vector<float>& perfectMatchVal)
{
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

    if ((strCompMethod == "emd") || (strCompMethod == "all"))
    {
        compMethods.push_back(CV_COMP_EMD);
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

    case CV_COMP_EMD:
        return "EMD";

    default:
        return "invalid";
    }
}

void Str2DistMethod(
    const string& strDistance,
    vector<int>& distances)
{
    distances.clear();

    if ((strDistance == "l1") || (strDistance == "all"))
    {
        distances.push_back(DIST_L1);
    }

    if ((strDistance == "l2") || (strDistance == "all"))
    {
        distances.push_back(DIST_L2);
    }

    if ((strDistance == "c") || (strDistance == "all"))
    {
        distances.push_back(DIST_C);
    }

    return;
}

string DistMethod2Str(const int distMethod)
{
    switch (distMethod)
    {
    case DIST_L1:
        return "Manhattan distance";

    case DIST_L2:
        return "Euclidean distance";

    case DIST_C:
        return "Checkboard distance";

    default:
        return "Unsupported or invalid distance";
    }
}

// Courtesy of https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv.
string CvType2Str(const int type)
{
    string typeStr("CV_");

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth )
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

// Modified based on https://stackoverflow.com/questions/33830208/how-to-find-the-pixel-value-that-corresponds-to-a-specific-number-of-pixels/33833349#33833349
void DrawHistogram(
    const Mat& hist,
    Mat& histogram,
    const int binSize = 10,
    const int scale = 1000)
{
    // Flattern the multi-dimensional matrix into a one-dimentional vector. Note that the original element type of hist is CV_32FC1
    // and we convert it into an integer.
    vector<int> vhist;
    int maxVal = 0;
    for (MatConstIterator_<float> itHist = hist.begin<float>(); itHist != hist.end<float>(); ++itHist)
    {
        vhist.push_back(static_cast<int>((*itHist)*scale));
        maxVal = max(maxVal, vhist.back());
    }

    int rows = maxVal + 10;
    int cols = vhist.size() * binSize;
    histogram = Mat(rows, cols, CV_8UC3);

    for (int vhistIndex = 0; vhistIndex < static_cast<int>(vhist.size()); vhistIndex++)
    {
        int h = rows - vhist[vhistIndex];
        rectangle(
            histogram,
            Point(vhistIndex*binSize, h),
            Point((vhistIndex + 1)*binSize - 1, rows),
            (vhistIndex % 2) ? Scalar(0, 100, 255) : Scalar(0, 0, 255),
            CV_FILLED);
    }
}

void CreateSignatureFromHistogram(
    const Mat& hist,
    Mat& sig)
{
    if (hist.cols == 1)
    {
        // One-dimensional histogram
        vector<Vec2f> sigv;
        for (int rowIndex = 0; rowIndex < hist.rows; rowIndex++)
        {
            float binVal = hist.at<float>(rowIndex, 0);
            if (binVal != 0)
            {
                sigv.push_back(Vec2f(binVal, static_cast<float>(rowIndex)));
            }
        }

        sig = Mat(sigv).clone().reshape(1);
    }
    else
    {
        // Two-dimensional histogram
        vector<Vec3f> sigv;
        for (int rowIndex = 0; rowIndex < hist.rows; rowIndex++)
        {
            for (int colIndex = 0; colIndex < hist.cols; colIndex++)
            {
                float binVal = hist.at<float>(rowIndex, colIndex);
                if (binVal != 0)
                {
                    sigv.push_back(Vec3f(binVal, static_cast<float>(rowIndex), static_cast<float>(colIndex)));
                }
            }
        }

        sig = Mat(sigv).clone().reshape(1);
    }

    return;
}

int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("image1", po::value<string>()->required(), "The first image")  // This is a positional option.
        ("image2", po::value<string>()->required(), "The second image") // This is also a positional option.
        ("help,h", "Display the help information")
        ("bgr-channels,c", po::value<string>(), "The BGR channels used for generating the histogram (b | g | r). If not specified, default bgr. Note that one histogram is generated for each specified channel.")
        ("comparison-method,m", po::value<string>(), "The comparison method (correl | chisqr | chisqr_alt | intersect | bhattacharyya | hellinger | kl_div | emd | all). If not specified, default correl.")
        ("distance,d", po::value<string>(), "The distance used by the EMD comparison (l1 | l2 | c | all). If not specified, default l1.");

    po::positional_options_description posOpt;
    posOpt.add("image1", 1);
    posOpt.add("image2", 1);

    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(opt).positional(posOpt).run(), vm);

        if (vm.count("help") > 0)
        {
            cout << "Usage: ./BgrHistComparison image1 image2 -c [BGR-channels] -m [comparison-method] -d [distance]" << endl << endl;
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
    string strBgrChannels;
    string strCompMethod;
    string strDistance;

    img1 = vm["image1"].as<string>();
    img2 = vm["image2"].as<string>();

    if (vm.count("bgr-channels") > 0)
    {
        strBgrChannels = vm["bgr-channels"].as<string>();
    }
    else
    {
        strBgrChannels = "bgr";
        cout << "[INFO]: no bgr channel is specified and use the default channels 0, 1, and 2 (i.e., b, g, and r)." << endl;
    }

    vector<int> bgrChannels;
    vector<int> histSize;
    vector<float> ranges;
    transform(strBgrChannels.begin(), strBgrChannels.end(), strBgrChannels.begin(), ::tolower);
    Str2BgrChannels(strBgrChannels, bgrChannels, histSize, ranges);
    if (bgrChannels.empty())
    {
        cerr << "[ERROR]: Invalid HSV channels " << strBgrChannels  << ", should contain h or s or v." << endl << endl;
        return -1;
    }

    if (vm.count("comparison-method") > 0)
    {
        strCompMethod = vm["comparison-method"].as<string>();
    }
    else
    {
        strCompMethod = "correl";
        cout << "[INFO]: No comparison method is specified and use the default comparison method Correlation." << endl;
    }

    vector<int> histComparisonMethods;
    vector<float> histCompPerfectMatchVals;
    transform(strCompMethod.begin(), strCompMethod.end(), strCompMethod.begin(), ::tolower);
    Str2HistComparisonMethod(strCompMethod, histComparisonMethods, histCompPerfectMatchVals);
    if (histComparisonMethods.empty())
    {
        cerr << "[ERROR]: Invalid histogram comparison method " << strCompMethod << "." << endl << endl;
        return -1;
    }

    vector<int> distances;
    if ((strCompMethod == "emd") || (strCompMethod == "all"))
    {
        if (vm.count("distance") > 0)
        {
            strDistance = vm["distance"].as<string>();
            transform(strDistance.begin(), strDistance.end(), strDistance.begin(), ::tolower);
        }
        else
        {
            strDistance = "l1";
            cout << "[INFO]: No distance type is specified and use the default L1 Manhattan distance." << endl;
        }

        Str2DistMethod(strDistance, distances);
        if (distances.empty())
        {
            cerr << "[ERROR]: Unsupported or invalid EMD distance method " << strDistance << "." << endl << endl;
            return -1;
        }
    }

    Mat srcImg1 = imread(img1, IMREAD_COLOR);
    if (srcImg1.empty())
    {
        cerr << "[ERROR]: Cannot load " << img1 << "." << endl << endl;
        return -1;
    }

    Mat srcImg2 = imread(img2, IMREAD_COLOR);
    if (srcImg2.empty())
    {
        cerr << "[ERROR]: Cannot load " << img2 << "." << endl << endl;
        return -1;
    }

    // Display the original image 1 and 2 in one window side by side.
    if (srcImg1.rows == srcImg2.rows)
    {
        Mat srcImgs;
        hconcat(srcImg1, Mat::zeros(srcImg1.rows, 10, CV_8UC3), srcImg1);
        hconcat(srcImg1, srcImg2, srcImgs);

        namedWindow("The original image 1 and 2", WINDOW_AUTOSIZE);
        imshow("The original image 1 and 2", srcImgs);
    }

    vector<Mat> hist1(bgrChannels.size());
    vector<Mat> hist2(bgrChannels.size());
    vector<Mat> histogram1(bgrChannels.size());
    vector<Mat> histogram2(bgrChannels.size());
    vector<Mat> histograms(bgrChannels.size());
    vector<Mat> sig1(bgrChannels.size());
    vector<Mat> sig2(bgrChannels.size());
    for (size_t channelIndex = 0; channelIndex < bgrChannels.size(); ++channelIndex)
    {
        cout << "============================================================================================" << endl;
        cout << "[INFO]: Result of channel " << BgrChannel2Str(bgrChannels[channelIndex]) << ":" << endl;

        // Compute the histograms of image 1 and 2, respectively.
        calcHist(vector<Mat>{srcImg1}, vector<int>{bgrChannels[channelIndex]}, noArray(), hist1[channelIndex], histSize, ranges);
        calcHist(vector<Mat>{srcImg2}, vector<int>{bgrChannels[channelIndex]}, noArray(), hist2[channelIndex], histSize, ranges);

        // Normalize the histograms.
        normalize(hist1[channelIndex], hist1[channelIndex], 1, 0, NORM_L1);
        normalize(hist2[channelIndex], hist2[channelIndex], 1, 0, NORM_L1);

        DrawHistogram(hist1[channelIndex], histogram1[channelIndex]);
        DrawHistogram(hist2[channelIndex], histogram2[channelIndex]);

        vconcat(histogram1[channelIndex], Mat::zeros(10, histogram1[channelIndex].cols, CV_8UC3), histogram1[channelIndex]);
        vconcat(histogram1[channelIndex], histogram2[channelIndex], histograms[channelIndex]);

        string windowTitle = "The histograms of channel " + BgrChannel2Str(bgrChannels[channelIndex]);
        namedWindow(windowTitle, WINDOW_AUTOSIZE);
        imshow(windowTitle, histograms[channelIndex]);

        // Compare the two normalized histograms.
        for (size_t compMethodIndex = 0; compMethodIndex < histComparisonMethods.size(); ++compMethodIndex)
        {
            double compareResult = 0;

            if (histComparisonMethods[compMethodIndex] != CV_COMP_EMD)
            {
                compareResult = compareHist(hist1[channelIndex], hist2[channelIndex], histComparisonMethods[compMethodIndex]);

                cout << "[INFO]: The comparison result of the two histograms = " << compareResult << " with the method "
                    << HistComparisonMethod2Str(histComparisonMethods[compMethodIndex]) << endl;
            }
            else
            {
                // It needs more efforts for comparing the two histograms via EMD.
                // Create the signatures from the histograms which will be used by EMD.
                CreateSignatureFromHistogram(hist1[channelIndex], sig1[channelIndex]);
                CreateSignatureFromHistogram(hist2[channelIndex], sig2[channelIndex]);

                for (const int& distMethod : distances)
                {
                    compareResult = EMD(sig1[channelIndex], sig2[channelIndex], distMethod);
                    cout << "[INFO]: The comparison result of the two histograms = " << compareResult << " with the method "
                        << HistComparisonMethod2Str(histComparisonMethods[compMethodIndex]) << " and the "
                        << DistMethod2Str(distMethod) << "." << endl;
                }
            }

            cout << "[INFO]: The result for a perfect match of the method " << HistComparisonMethod2Str(histComparisonMethods[compMethodIndex])
                << " is " << histCompPerfectMatchVals[compMethodIndex] << "." << endl;
        }

        cout << "============================================================================================" << endl;
    }

    /*
    // Compare the two normalized histograms.
    for (size_t compMethodIndex = 0; compMethodIndex < histComparisonMethods.size(); ++compMethodIndex)
    {
        double compareResult = compareHist(hist1, hist2, histComparisonMethods[compMethodIndex]);
        cout << "[INFO]: The comparison result of the two histograms = " << compareResult << " with the method "
            << HistComparisonMethod2Str(histComparisonMethods[compMethodIndex]) << endl;
        cout << "[INFO]: The result for a perfect match of the method " << HistComparisonMethod2Str(histComparisonMethods[compMethodIndex])
            << " is " << histCompPerfectMatchVals[compMethodIndex] << "." << endl;
    }
    */

    waitKey();

    // Destroy the windows.
    for (const int& channel : bgrChannels)
    {
        string windowTitle = "The histograms of channel " + BgrChannel2Str(channel);
        destroyWindow(windowTitle);
    }

    if (srcImg1.rows == srcImg2.rows)
    {
        destroyWindow("The original image 1 and 2");
    }

    return 0;
}
