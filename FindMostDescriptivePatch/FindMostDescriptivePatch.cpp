/*
 * FindMostDescriptivePatch.cpp
 *
 *  Created on: Aug 2, 2017
 *      Author: renwei
 */

#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>

#include <boost/program_options.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
namespace po = boost::program_options;

typedef std::chrono::high_resolution_clock Clock;

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

int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("srcImg", po::value<string>()->required(), "The input source image")  // This is a positional option.
        ("charPatch", po::value<string>()->required(), "The output characteristic patch") // This is also a positional option.
        ("help,h", "Display the help information")
        ("width,w", po::value<int>(), "The width of the characteristic patch")
        ("height,g", po::value<int>(), "The height of the characteristic patch")
        ("buffer-width,b", po::value<int>(), "The width of the buffer zone in which the keypoints are not counted")
        ("buffer-height,d", po::value<int>(), "The height of the buffer zone in which the keypoints are not counted");

    po::positional_options_description posOpt;
    posOpt.add("srcImg", 1);
    posOpt.add("charPatch", 1);

    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(opt).positional(posOpt).run(), vm);

        if (vm.count("help") > 0)
        {
            cout << "Usage: ./FindMostDescriptivePath srcImg charPatch -w [patch-width] -g [patch-height] -b [buffer-width]" << endl << endl;
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
    string charPatchFile;
    int charPatchWidth = 0;
    int charPatchHeight = 0;
    int bufferWidth = 0;
    int bufferHeight = 0;

    // Read the options specified in the command.
    srcImgFile = vm["srcImg"].as<string>();
    charPatchFile = vm["charPatch"].as<string>();

    if (vm.count("width") > 0)
    {
        charPatchWidth = vm["width"].as<int>();
    }
    else
    {
        charPatchWidth = 100;
        cout << "[INFO]: The width of the characteristic patch is not specified and use the default value 100." << endl;
    }

    if (vm.count("height") > 0)
    {
        charPatchHeight = vm["height"].as<int>();
    }
    else
    {
        charPatchHeight = 100;
        cout << "[INFO]: The height of the characteristic patch is not specified and use the default value 100." << endl;
    }

    if (vm.count("buffer-width") > 0)
    {
        bufferWidth = vm["buffer-width"].as<int>();
    }
    else
    {
        bufferWidth = 10;
        cout << "[INFO]: The width of the buffer zone is not specified and use the default value 10." << endl;
    }

    if (vm.count("buffer-height") > 0)
    {
        bufferHeight = vm["buffer-height"].as<int>();
    }
    else
    {
        bufferHeight = 10;
        cout << "[INFO]: The height of the buffer zone is not specified and use the default value 10." << endl;
    }

    // Load the source image.
    Mat srcImg = imread(srcImgFile, IMREAD_COLOR);
    if (srcImg.empty())
    {
        cerr << "[ERROR]: Cannot load the source image " << srcImgFile << "." << endl << endl;
        return -1;
    }

    // Validate the options.
    if (charPatchWidth > srcImg.cols)
    {
        cerr << "[ERROR]: The width " << charPatchWidth << " of the characteristic patch is larger than that "
            << srcImg.cols << " of the source image." << endl << endl;
        return -1;
    }

    if (charPatchHeight > srcImg.rows)
    {
        cerr << "[ERROR]: The height " << charPatchHeight << " of the characteristic patch is larger than that "
            << srcImg.rows << " of the source image." << endl << endl;
        return -1;
    }

    if (2*bufferWidth >= charPatchWidth)
    {
        cerr << "[ERROR]: The width " << bufferWidth << " of the buffer zone is larger than or equal to 1/2 of that "
            << charPatchWidth << " of the characteristic patch." << endl << endl;
        return -1;
    }

    if (2*bufferHeight >= charPatchHeight)
    {
        cerr << "[ERROR]: The height " << bufferHeight << " of the buffer zone is larger than or equal to 1/2 of that "
            << charPatchHeight << " of the characteristic patch." << endl << endl;
        return -1;
    }

    // Detect the keypoints using the SURF detector and compute the descriptors which describe the keypoints.
    const int minHessian = 400;
    Ptr<SURF> surfDetector = SURF::create(minHessian);

    vector<KeyPoint> keypoints;
    Mat descriptors;

    auto tStart = Clock::now();
    surfDetector->detectAndCompute(srcImg, noArray(), keypoints, descriptors);
    auto tSurfEnd = Clock::now();

    cout << "[INFO]: Detect " << keypoints.size() << " keypoints and compute their descriptors in "
         << chrono::duration_cast<chrono::milliseconds>(tSurfEnd - tStart).count() << "ms." << endl;

    Point topLeftCorner(0, 0);
    if (!keypoints.empty())
    {
        Mat kpCnts(srcImg.rows - charPatchHeight + 1, srcImg.cols - charPatchWidth + 1, CV_32SC1, Scalar(0));

        for (const auto& kp : keypoints)
        {
            int leftMost = max(static_cast<int>(kp.pt.x) - charPatchWidth + bufferWidth + 1, 0);
            int rightMost = min(static_cast<int>(kp.pt.x) - bufferWidth + 1, srcImg.cols - charPatchWidth);
            int topMost = max(static_cast<int>(kp.pt.y) - charPatchHeight + bufferHeight + 1, 0);
            int bottomMost = min(static_cast<int>(kp.pt.y) - bufferHeight + 1, srcImg.rows - charPatchHeight);

            for (int rowIndex = topMost; rowIndex <= bottomMost; ++rowIndex)
            {
                for (int colIndex = leftMost; colIndex <= rightMost; ++colIndex)
                {
                    ++(kpCnts.at<int>(rowIndex, colIndex));
                }
            }
        }

        // Slower method: Slide the patch over the source image and find the one with the most keypoints in
        // its internal area (not in its buffer zone).
        //for (int rowIndex = 0; rowIndex < kpCnts.rows; ++rowIndex)
        //{
        //    for (int colIndex = 0; colIndex < kpCnts.cols; ++colIndex)
        //    {
        //        for (const auto& kp : keypoints)
        //        {
        //            if ((static_cast<int>(kp.pt.x) >= colIndex + bufferWidth) &&
        //                (static_cast<int>(kp.pt.x) < colIndex + charPatchWidth - bufferWidth) &&
        //                (static_cast<int>(kp.pt.y) >= rowIndex + bufferHeight) &&
        //                (static_cast<int>(kp.pt.y) < rowIndex + charPatchHeight - bufferHeight))
        //            {
        //                ++(kpCnts.at<int>(rowIndex, colIndex));
        //            }
        //        }
        //    }
        //}

        double maxVal = 0.0;
        minMaxLoc(kpCnts, nullptr, &maxVal, nullptr, &topLeftCorner);

        auto tEnd = Clock::now();

        cout << "[INFO]: Find the top left corner of the characteristic patch at Point (" << topLeftCorner.x << ", "
            << topLeftCorner.y << ") in " << chrono::duration_cast<chrono::milliseconds>(tEnd - tSurfEnd).count()
            << "ms." << endl;
    }
    else
    {
        // Since no keypoints are found, simply select the center patch as the result.
        cout << "[INFO]: No keypoints are found and select the center patch as the result." << endl;
        topLeftCorner.x = (srcImg.cols - charPatchWidth)/2;
        topLeftCorner.y = (srcImg.rows - charPatchHeight)/2;
    }

    // Write the characteristic patch into the file given by the option "charPatch".
    Mat charPatch = srcImg(Rect(topLeftCorner.x, topLeftCorner.y, charPatchWidth, charPatchHeight));
    imwrite(charPatchFile, charPatch);

    // Display the source image with all its keypoints and mark the characteristic patch by a black rectangle.
    drawKeypoints(srcImg, keypoints, srcImg);

    rectangle(
        srcImg,
        topLeftCorner,
        Point(topLeftCorner.x + charPatchWidth, topLeftCorner.y + charPatchHeight),
        Scalar::all(0),
        2,
        8,
        0);

    namedWindow("Source image", WINDOW_AUTOSIZE);
    imshow("Source image", srcImg);

    // Display the characteristic patch with the keypoints.
    namedWindow("Patch", WINDOW_AUTOSIZE);
    imshow("Patch", charPatch);

    waitKey();

    destroyWindow("Source image");
    destroyWindow("Patch");

    return 0;
}
