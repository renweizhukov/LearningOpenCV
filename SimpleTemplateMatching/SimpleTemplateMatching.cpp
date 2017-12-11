/*
 * SimpleTemplateMatching.cpp
 *
 *  Created on: Jul 27, 2017
 *      Author: renwei
 *
 * Modified based on the example given at http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
 *
 */

#include <iostream>
#include <string>
#include <algorithm>

#include <boost/program_options.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
namespace po = boost::program_options;

enum class TemplateMatchingColorChannels
{
    BGR = 0,
    GRAYSCALE,
    HSV,
    HS,
    INVALID
};

// Global variables
Mat originalSrcImg;
Mat originalTemplImg;
Mat srcImg;
Mat templImg;
Mat result;
string imgWindow("Source image");
string resWindow("Result window");

int matchMethod;
const int maxTrackbar = 5;

TemplateMatchingColorChannels Str2TemplateMatchingColorChannels(string& strColorChannels)
{
    transform(strColorChannels.begin(), strColorChannels.end(), strColorChannels.begin(), ::tolower);

    if (strColorChannels == "bgr")
    {
        return TemplateMatchingColorChannels::BGR;
    }
    else if (strColorChannels == "grayscale")
    {
        return TemplateMatchingColorChannels::GRAYSCALE;
    }
    else if (strColorChannels == "hsv")
    {
        return TemplateMatchingColorChannels::HSV;
    }
    else if (strColorChannels == "hs")
    {
        return TemplateMatchingColorChannels::HS;
    }
    else
    {
        return TemplateMatchingColorChannels::INVALID;
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

string TemplateMatchingMethod2Str(const int matchingMethod)
{
    switch (matchingMethod)
    {
    case TM_SQDIFF:
        return "Square Difference";

    case TM_SQDIFF_NORMED:
        return "Normalized Square Difference";

    case TM_CCORR:
        return "Cross Correlation";

    case TM_CCORR_NORMED:
        return "Normalized Cross Correlation";

    case TM_CCOEFF:
        return "Correlation Coefficient";

    case TM_CCOEFF_NORMED:
        return "Normalized Correlation Coefficient";

    default:
        return "Invalid";
    }
}

void ConvertImgForTemplateMatching(
     const Mat& srcImg,
     Mat& desImg,
     const TemplateMatchingColorChannels colorChannels)
{
    if (colorChannels == TemplateMatchingColorChannels::BGR)
    {
        srcImg.copyTo(desImg);
    }
    else if (colorChannels == TemplateMatchingColorChannels::GRAYSCALE)
    {
        cvtColor(srcImg, desImg, COLOR_BGR2GRAY);
    }
    else
    {
        // colorChannels is either HSV or HS.
        cvtColor(srcImg, desImg, COLOR_BGR2HSV);

        if (colorChannels == TemplateMatchingColorChannels::HS)
        {
            // Remove the last channel "Value".
            Mat hsImg(desImg.rows, desImg.cols, CV_8UC2);
            int fromToMap[4] = {0, 0, 1, 1};

            mixChannels(vector<Mat>{desImg}, vector<Mat>{hsImg}, fromToMap, 2);

            hsImg.copyTo(desImg);
        }
    }
}

/**
 * @function MatchingMethod
 * @brief Trackbar callback
 */
void MatchingMethod(int, void*)
{
    // Source image to display
    Mat imgDisplay;
    originalSrcImg.copyTo(imgDisplay);

    // Create the result matrix
    const int resultRows = srcImg.rows - templImg.rows + 1;
    const int resultCols =  srcImg.cols - templImg.cols + 1;

    result.create(resultRows, resultCols, CV_32FC1);

    //cout << "[DEBUG]: srcImg #rows = " << srcImg.rows << ", #cols = " << srcImg.cols << "." << endl;
    //cout << "[DEBUG]: templImg #rows = " << templImg.rows << ", #cols = " << templImg.cols << "." << endl;

    // Do the Template Matching.
    matchTemplate(srcImg, templImg, result, matchMethod);

    //cout << "[DEBUG]: result #rows = " << result.rows << ", #cols = " << result.cols << "." << endl;

    // Localize the best match with minMaxLoc
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;

    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better.
    // Note that since result has been normalized into the range [0, 1], the best value is either 0 or 1.
    if ((matchMethod == CV_TM_SQDIFF) || (matchMethod == CV_TM_SQDIFF_NORMED))
    {
        matchLoc = minLoc;
        cout << "[INFO]: Template matching minVal = " << minVal << "." << endl;
    }
    else
    {
        matchLoc = maxLoc;
        cout << "[INFO]: Template matching maxVal = " << maxVal << "." << endl;
    }

    // Show me what you got
    rectangle(imgDisplay, matchLoc, Point(matchLoc.x + templImg.cols, matchLoc.y + templImg.rows), Scalar(0, 0, 255), 2, 8, 0);
    rectangle(result, matchLoc, Point(matchLoc.x + templImg.cols , matchLoc.y + templImg.rows), Scalar(255, 0, 0), 2, 8, 0);

    imshow(imgWindow, imgDisplay);
    imshow(resWindow, result);

    return;
}

/** @function main */
int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("srcImg", po::value<string>()->required(), "The source image")  // This is a positional option.
        ("templImg", po::value<string>()->required(), "The template image") // This is also a positional option.
        ("help,h", "Display the help information")
        ("color-channels,c", po::value<string>(), "The color channels used by the template matching (bgr | grayscale | hsv | hs). If not specified, default bgr.");

    po::positional_options_description posOpt;
    posOpt.add("srcImg", 1);
    posOpt.add("templImg", 1);

    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(opt).positional(posOpt).run(), vm);

        if (vm.count("help") > 0)
        {
            cout << "Usage: ./SimpleTemplateMatching srcImg templImg -c [Color-channels]" << endl << endl;
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

    string srcImgFile;
    string templImgFile;
    string strColorChannels;
    TemplateMatchingColorChannels colorChannels = TemplateMatchingColorChannels::INVALID;

    srcImgFile = vm["srcImg"].as<string>();
    templImgFile = vm["templImg"].as<string>();

    if (vm.count("color-channels") > 0)
    {
        strColorChannels = vm["color-channels"].as<string>();
        colorChannels = Str2TemplateMatchingColorChannels(strColorChannels);
    }
    else
    {
        colorChannels = TemplateMatchingColorChannels::BGR;
        cout << "[INFO]: No color channel is specified and use the default BGR channels for template matching." << endl;
    }

    if (colorChannels == TemplateMatchingColorChannels::INVALID)
    {
        cerr << "[ERROR]: Invalid color channels = " << strColorChannels << "." << endl << endl;
        return -1;
    }

    // Load the source image and the template image.
    originalSrcImg = imread(srcImgFile, IMREAD_COLOR);
    if (originalSrcImg.empty())
    {
        cerr << "[ERROR]: Cannot load the source image " << srcImgFile << "." << endl << endl;
        return -1;
    }
    
    originalTemplImg = imread(templImgFile, IMREAD_COLOR);
    if (originalTemplImg.empty())
    {
        cerr << "[ERROR]: Cannot load the template image " << templImgFile << "." << endl << endl;
        return -1;
    }

    cout << "[DEBUG]: Original source image data type = " << CvType2Str(originalSrcImg.type()) << "." << endl;
    cout << "[DEBUG]: Original template image data type = " << CvType2Str(originalTemplImg.type()) << "." << endl;

    ConvertImgForTemplateMatching(originalSrcImg, srcImg, colorChannels);
    ConvertImgForTemplateMatching(originalTemplImg, templImg, colorChannels);

    cout << "[DEBUG]: Source image data type = " << CvType2Str(srcImg.type()) << " after the conversion for template matching." << endl;
    cout << "[DEBUG]: Template image data type = " << CvType2Str(templImg.type()) << " after the conversion for template matching." << endl;

    // Create windows
    namedWindow(imgWindow, CV_WINDOW_AUTOSIZE);
    namedWindow(resWindow, CV_WINDOW_AUTOSIZE);

    // Create Trackbar
    const string trackbarLabel("Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED");
    createTrackbar(trackbarLabel, imgWindow, &matchMethod, maxTrackbar, MatchingMethod);

    MatchingMethod(0, nullptr);

    waitKey(0);

    destroyWindow(resWindow);
    destroyWindow(imgWindow);

    return 0;
}
