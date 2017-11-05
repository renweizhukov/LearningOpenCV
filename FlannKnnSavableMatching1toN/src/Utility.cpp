/*
 * Utility.cpp
 *
 *  Created on: Aug 15, 2017
 *      Author: renwei
 */

#include "Utility.h"

using namespace std;
using namespace cv;

int Utility::LoadAllImagesInDir(
        const string& imgDir,
        vector<string>& imgFilenameList,
        vector<Mat>& imgs)
{
    DIR *dp = nullptr;
    struct dirent *dirp = nullptr;

    dp = opendir(imgDir.c_str());
    if (dp == nullptr)
    {
        cerr << "[ERROR]: opendir failed to open " << imgDir
            << " with error " << strerror(errno) << "." << endl << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != nullptr)
    {
        // Find all the files which share the common prefix in their names.
        if (dirp->d_type == DT_REG)
        {
            string imgFullname = imgDir + '/' + string(dirp->d_name);
            Mat img = imread(imgFullname);

            if (!img.empty())
            {
                imgFilenameList.push_back(dirp->d_name);
                imgs.push_back(img);
                cout << "[INFO]: Loaded image " << dirp->d_name << "." << endl;
            }
            else
            {
                cerr << "[WARNING]: " << dirp->d_name << " is not an image file." << endl;
            }
        }
    }

    cout << "[INFO]: Loaded " << imgFilenameList.size() << " images in " << imgDir << "." << endl;

    closedir(dp);
    return 0;
}

void Utility::SeparateDirFromFilename(
    const string& fullFilename,
    string& dir,
    string& filename)
{
    dir = "./";
    filename = fullFilename;

    size_t posLastSlash = fullFilename.find_last_of('/');
    size_t posLastDot = fullFilename.find_last_of('.');
    if (posLastSlash != string::npos)
    {
        dir = fullFilename.substr(0, posLastSlash + 1);
        if (posLastDot > posLastSlash)
        {
            filename = fullFilename.substr(posLastSlash + 1, posLastDot - posLastSlash - 1);
        }
        else
        {
            filename = fullFilename.substr(posLastSlash + 1);
        }
    }
}

// Courtesy of https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv.
string Utility::CvType2Str(const int type)
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

void Utility::FilterKeypointsAndDescriptors(
        const cv::Rect2f& rect,
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Mat& descriptors,
        std::vector<cv::KeyPoint>& filteredKeypoints,
        cv::Mat& filteredDescriptors)
{
    for (size_t keypointIndex = 0; keypointIndex < keypoints.size(); ++keypointIndex)
    {
        // Keep the keypoint and the corresponding descriptor if the keypoint
        // is located within the rectangle, otherwise throw them away.
        if (rect.contains(keypoints[keypointIndex].pt))
        {
            filteredKeypoints.push_back(keypoints[keypointIndex]);
            filteredDescriptors.push_back(descriptors.row(keypointIndex));
        }
    }

    cout << "[INFO]: Keep " << filteredDescriptors.rows << " descriptors out of "
        << descriptors.rows << "." << endl;
}
