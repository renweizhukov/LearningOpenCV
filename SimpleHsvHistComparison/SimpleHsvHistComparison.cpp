/*
 * SimpleHsvHistComparison.cpp
 *
 *  Created on: Jul 21, 2017
 *      Author: renwei
 */

// Modified based on Example 13-1: Histogram computation and display in the book "Learning OpenCV 3".

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

int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("image1", po::value<string>()->required(), "The first image")  // This is a positional option.
        ("image2", po::value<string>()->required(), "The second image") // This is also a positional option.
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
            cout << "Usage: ./SimpleHsvHistComparison image1 image2 -m [comparison-method] -c [HSV-channels]" << endl << endl;
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
    string strCompMethod;
    string strHsvChannels;

    img1 = vm["image1"].as<string>();
    img2 = vm["image2"].as<string>();

    if (vm.count("comparison-method") > 0)
    {
        strCompMethod = vm["comparison-method"].as<string>();
    }
    else
    {
        strCompMethod = "correl";
        cout << "[INFO]: No distance method is specified and use the default distance method Correlation." << endl;
    }

    vector<int> histComparisonMethods;
    vector<float> histCompPerfectMatchVals;
    Str2HistComparisonMethod(strCompMethod, histComparisonMethods, histCompPerfectMatchVals);
    if (histComparisonMethods.empty())
    {
        cerr << "[ERROR]: Invalid histogram comparison method " << strCompMethod << "." << endl << endl;
        return -1;
    }

    if (vm.count("hsv-channels") > 0)
    {
        strHsvChannels = vm["hsv-channels"].as<string>();
    }
    else
    {
        strHsvChannels = "hs";
        cout << "[INFO]: no hsv channel is specified and use the default channels 0 and 1 (i.e., h and s)." << endl;
    }

    vector<int> hsvChannels;
    vector<int> histSize;
    vector<float> ranges;
    Str2HsvChannels(strHsvChannels, hsvChannels, histSize, ranges);
    if (hsvChannels.empty())
    {
        cerr << "[ERROR]: Invalid HSV channels " << strHsvChannels  << ", should contain h or s or v." << endl << endl;
        return -1;
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
    
    // Convert image 1 and 2 into HSV.
    Mat hsvImg1;
    Mat hsvImg2;
    cvtColor(srcImg1, hsvImg1, COLOR_BGR2HSV);
    cvtColor(srcImg2, hsvImg2, COLOR_BGR2HSV);

    // Compute the histograms of image 1 and 2, respectively.
    Mat hist1;
    Mat hist2;
    calcHist(vector<Mat>{hsvImg1}, hsvChannels, noArray(), hist1, histSize, ranges);
    calcHist(vector<Mat>{hsvImg2}, hsvChannels, noArray(), hist2, histSize, ranges);

    // Normalize the histograms.
    normalize(hist1, hist1, 1, 0, NORM_L1);
    normalize(hist2, hist2, 1, 0, NORM_L1);

    if ((hist1.cols == 1) && (hist2.cols == 1))
    {
        Mat histogram1;
        Mat histogram2;

        DrawHistogram(hist1, histogram1);
        DrawHistogram(hist2, histogram2);

        Mat histograms;
        vconcat(histogram1, Mat::zeros(10, histogram1.cols, CV_8UC3), histogram1);
        vconcat(histogram1, histogram2, histograms);

        namedWindow("The histograms of image 1 and 2", WINDOW_AUTOSIZE);
        imshow("The histograms of image 1 and 2", histograms);
    }
    else
    {
        cout << "[INFO]: Can't draw the histograms for more than one HSV channel." << endl;
    }

    // Compare the two normalized histograms.
    for (size_t compMethodIndex = 0; compMethodIndex < histComparisonMethods.size(); ++compMethodIndex)
    {
        double compareResult = compareHist(hist1, hist2, histComparisonMethods[compMethodIndex]);
        cout << "[INFO]: The comparison result of the two histograms = " << compareResult << " with the method "
            << HistComparisonMethod2Str(histComparisonMethods[compMethodIndex]) << endl;
        cout << "[INFO]: The result for a perfect match of the method " << HistComparisonMethod2Str(histComparisonMethods[compMethodIndex])
            << " is " << histCompPerfectMatchVals[compMethodIndex] << "." << endl;
    }

    waitKey();

    // Destroy the windows.
    if ((hist1.dims == 1) && (hist2.dims == 1))
    {
        destroyWindow("The histograms of image 1 and 2");
    }
    
    if (srcImg1.rows == srcImg2.rows)
    {
        destroyWindow("The original image 1 and 2");
    }

    return 0;
}
