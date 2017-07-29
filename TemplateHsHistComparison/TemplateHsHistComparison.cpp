/*
 * TemplateHsHistComparison.cpp
 *
 *  Created on: Jul 28, 2017
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
    string& strDistance,
    vector<int>& distances)
{
    transform(strDistance.begin(), strDistance.end(), strDistance.begin(), ::tolower);

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

void ConvertImgForTemplateMatching(
     const Mat& srcImg,
     Mat& desImg,
     Mat& desImgHs)
{
    // colorChannels is either HSV or HS.
    cvtColor(srcImg, desImg, COLOR_BGR2HSV);

    // Remove the last channel "Value".
    desImgHs.create(desImg.rows, desImg.cols, CV_8UC2);
    int fromToMap[4] = {0, 0, 1, 1};

    mixChannels(vector<Mat>{desImg}, vector<Mat>{desImgHs}, fromToMap, 2);
}

Point GetTemplateMatchingPoint(
    const Mat& srcImg,
    const Mat& templImg,
    Mat& result)
{
    // Create the result matrix.
    const int resultRows = srcImg.rows - templImg.rows + 1;
    const int resultCols =  srcImg.cols - templImg.cols + 1;

    result.create(resultRows, resultCols, CV_32FC1);

    // Do the Template Matching and Normalize.
    matchTemplate(srcImg, templImg, result, TM_CCOEFF_NORMED);
    normalize(result, result, 0, 1, NORM_MINMAX);

    // Localize the best match with minMaxLoc.
    double maxVal;
    Point maxLoc;

    minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);

    // For CCOEFF_NORMED, the best match is the maximum value.
    // Note that since result has been normalized into the range [0, 1], the maximum value is 1.
    return maxLoc;
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
        ("srcImg", po::value<string>()->required(), "The source image")  // This is a positional option.
        ("templImg", po::value<string>()->required(), "The template image") // This is also a positional option.
        ("help,h", "Display the help information")
        ("comparison-method,m", po::value<string>(), "The comparison method (correl | chisqr | chisqr_alt | intersect | bhattacharyya | hellinger | kl_div | emd | all). If not specified, default correl.")
        ("distance,d", po::value<string>(), "The distance used by the EMD comparison (l1 | l2 | c | all). If not specified, default l1.");

    po::positional_options_description posOpt;
    posOpt.add("srcImg", 1);
    posOpt.add("templImg", 1);

    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(opt).positional(posOpt).run(), vm);

        if (vm.count("help") > 0)
        {
            cout << "Usage: ./TemplateHsHistComparison srcImg templImg -m [comparison-method] -d [distance]" << endl << endl;
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
    string strCompMethod;
    string strDistance;

    srcImgFile = vm["srcImg"].as<string>();
    templImgFile = vm["templImg"].as<string>();

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

    // Load the source image and the template image.

    Mat originalSrcImg = imread(srcImgFile, IMREAD_COLOR);
    if (originalSrcImg.empty())
    {
        cerr << "[ERROR]: Cannot load the source image " << srcImgFile << "." << endl << endl;
        return -1;
    }

    Mat originalTemplImg = imread(templImgFile, IMREAD_COLOR);
    if (originalTemplImg.empty())
    {
        cerr << "[ERROR]: Cannot load the template image " << templImgFile << "." << endl << endl;
        return -1;
    }

    // Convert the source and template images into HSV and then delete the Value channel.
    Mat srcImg;
    Mat templImg;
    Mat srcImgHs;
    Mat templImgHs;
    ConvertImgForTemplateMatching(originalSrcImg, srcImg, srcImgHs);
    ConvertImgForTemplateMatching(originalTemplImg, templImg, templImgHs);

    cout << "[DEBUG]: Source image data type = " << CvType2Str(srcImgHs.type()) << " after the conversion for template matching." << endl;
    cout << "[DEBUG]: Template image data type = " << CvType2Str(templImgHs.type()) << " after the conversion for template matching." << endl;

    // Display the template image.
    const string templImgWindow("The template image");
    namedWindow(templImgWindow, WINDOW_AUTOSIZE);
    imshow(templImgWindow, originalTemplImg);

    // Do the template matching and find the best match point.
    Mat result;
    Point matchPoint = GetTemplateMatchingPoint(srcImgHs, templImgHs, result);

    //Mat bestMatchedPatchRgb = originalSrcImg(Rect(matchPoint.x, matchPoint.y, templImg.cols, templImg.rows));
    //namedWindow("The best matched patch", WINDOW_AUTOSIZE);
    //imshow("The best matched patch", bestMatchedPatchRgb);

    // Display the original source image in one window and mark the best matched patch by a black rectangle.
    // Display the result in another window.
    rectangle(
        originalSrcImg,
        matchPoint,
        Point(matchPoint.x + templImg.cols, matchPoint.y + templImg.rows),
        Scalar::all(0),
        2,
        8,
        0);
    rectangle(
        result,
        matchPoint,
        Point(matchPoint.x + templImg.cols , matchPoint.y + templImg.rows),
        Scalar::all(0),
        2,
        8,
        0);

    const string srcImgWindow("The source image");
    namedWindow(srcImgWindow, WINDOW_AUTOSIZE);
    imshow(srcImgWindow, originalSrcImg);

    const string resWindow("The template matching result");
    namedWindow(resWindow, WINDOW_AUTOSIZE);
    imshow(resWindow, result);

    // Crop the patch of the source image which best matches the template image.
    Mat bestMatchedPatch = srcImg(Rect(matchPoint.x, matchPoint.y, templImg.cols, templImg.rows));

    Mat bestMatchedPatchHist;
    Mat templHist;

    vector<int> hsChannels{0, 1};   // Channel hue and saturation
    vector<int> histSize{30, 32};   // 30 bins for hue and 32 bins for saturation
    vector<float> ranges{0, 180, 0, 256};   // The range of hue is [0, 180) and the range of saturation is [0, 256).

    // Compute the histograms of the best-matched patch and the template image, respectively.
    calcHist(vector<Mat>{bestMatchedPatch}, hsChannels, noArray(), bestMatchedPatchHist, histSize, ranges);
    calcHist(vector<Mat>{templImg}, hsChannels, noArray(), templHist, histSize, ranges);

    // Normalize the histograms.
    normalize(bestMatchedPatchHist, bestMatchedPatchHist, 1, 0, NORM_L1);
    normalize(templHist, templHist, 1, 0, NORM_L1);

    // Compare the two normalized histograms.
    for (size_t compMethodIndex = 0; compMethodIndex < histComparisonMethods.size(); ++compMethodIndex)
    {
        double compareResult = 0;

        if (histComparisonMethods[compMethodIndex] != CV_COMP_EMD)
        {
            compareResult = compareHist(bestMatchedPatchHist, templHist, histComparisonMethods[compMethodIndex]);

            cout << "[INFO]: The comparison result of the two histograms = " << compareResult << " with the method "
                << HistComparisonMethod2Str(histComparisonMethods[compMethodIndex]) << endl;
        }
        else
        {
            // It needs more efforts for comparing the two histograms via EMD.
            // Create the signatures from the histograms which will be used by EMD.
            Mat bestMatchedPatchSig;
            Mat templSig;

            CreateSignatureFromHistogram(bestMatchedPatchHist, bestMatchedPatchSig);
            CreateSignatureFromHistogram(templHist, templSig);

            for (const int& distMethod : distances)
            {
                compareResult = EMD(bestMatchedPatchSig, templSig, distMethod);
                cout << "[INFO]: The comparison result of the two histograms = " << compareResult << " with the method "
                    << HistComparisonMethod2Str(histComparisonMethods[compMethodIndex]) << " and the "
                    << DistMethod2Str(distMethod) << "." << endl;
            }
        }

        cout << "[INFO]: The result for a perfect match of the method " << HistComparisonMethod2Str(histComparisonMethods[compMethodIndex])
            << " is " << histCompPerfectMatchVals[compMethodIndex] << "." << endl;
    }

    waitKey();

    destroyWindow(resWindow);
    destroyWindow(srcImgWindow);
    destroyWindow(templImgWindow);

    return 0;
}
