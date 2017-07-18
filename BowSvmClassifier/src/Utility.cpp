/*
 * Utility.cpp
 *
 *  Created on: Jul 10, 2017
 *      Author: renwei
 */

#include "Utility.h"

using namespace std;
using namespace cv;

int Utility::GetImageLabels(
    const string& basePath,
    vector<string>& imgLabels)
{
    DIR *dp = nullptr;
    struct dirent *dirp = nullptr;

    dp = opendir(basePath.c_str());
    if (dp == nullptr)
    {
        cerr << "[ERROR]: opendir failed to open " << basePath
            << " with error " << strerror(errno) << "." << endl << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != nullptr)
    {
        // All the images with the same label are stored in a subdirectory of
        // the base path with the name "label". Note that we need to exclude
        // the trivial current directory "." and parent directory "..".
        if ((dirp->d_type == DT_DIR) &&
            (0 != strcmp(dirp->d_name, ".")) &&
            (0 != strcmp(dirp->d_name, "..")))
        {
            imgLabels.push_back(dirp->d_name);
            cout << "[INFO]: Found label " << dirp->d_name << "." << endl;
        }
    }

    cout << "[INFO]: Found " << imgLabels.size() << " total labels." << endl;

    closedir(dp);
    return 0;
}

int Utility::GetImagesWithLabels(
    const string& basePath,
    vector<pair<string, string> >& imgsWithLabels)
{
    vector<string> imgLabels;

    int error = GetImageLabels(basePath, imgLabels);
    if (error != 0)
    {
        return error;
    }

    DIR *dp = nullptr;
    struct dirent *dirp = nullptr;

    imgsWithLabels.clear();

    for (auto& label:imgLabels)
    {
        int cntImgs = 0;
        string labelPath = basePath + "/" + label;

        dp = opendir(labelPath.c_str());
        if (dp == nullptr)
        {
            cerr << "[ERROR]: opendir failed to open " << labelPath
                << " with error " << strerror(errno) << "." << endl << endl;
            return errno;
        }

        while ((dirp = readdir(dp)) != nullptr)
        {
            if (dirp->d_type == DT_REG)
            {
                imgsWithLabels.push_back(make_pair(label, string(dirp->d_name)));
                cntImgs++;
            }
        }

        cout << "[INFO]: Found " << cntImgs << " images with label " << label << "." << endl;

        closedir(dp);
        dp = nullptr;
    }

    return 0;
}

int Utility::GetLabelImageMap(
    const string& basePath,
    map<string, vector<string> >& label2ImgMap)
{
    vector<string> imgLabels;

    int error = GetImageLabels(basePath, imgLabels);
    if (error != 0)
    {
        return error;
    }

    DIR *dp = nullptr;
    struct dirent *dirp = nullptr;

    label2ImgMap.clear();

    for (auto& label:imgLabels)
    {
        string labelPath = basePath + "/" + label;

        dp = opendir(labelPath.c_str());
        if (dp == nullptr)
        {
            cerr << "[ERROR]: opendir failed to open " << labelPath
                << " with error " << strerror(errno) << "." << endl << endl;
            return errno;
        }

        while ((dirp = readdir(dp)) != nullptr)
        {
            if (dirp->d_type == DT_REG)
            {
                auto itMap = label2ImgMap.find(label);
                if (itMap == label2ImgMap.end())
                {
                    label2ImgMap.insert(make_pair(label, vector<string>{string(dirp->d_name)}));
                }
                else
                {
                    itMap->second.push_back(dirp->d_name);
                }
            }
        }

        cout << "[INFO]: Found " << label2ImgMap.count(label) << " images with label " << label << "." << endl;

        closedir(dp);
        dp = nullptr;
    }

    return 0;
}

int Utility::GetFilesWithCommonPrefix(
    const string& path,
    const string& commonPrefix,
    vector<string>& fileList)
{
    DIR *dp = nullptr;
    struct dirent *dirp = nullptr;

    dp = opendir(path.c_str());
    if (dp == nullptr)
    {
        cerr << "[ERROR]: opendir failed to open " << path
            << " with error " << strerror(errno) << "." << endl << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != nullptr)
    {
        // Find all the files which share the common prefix in their names.
        if (dirp->d_type == DT_REG)
        {
            string filePrefix = string(dirp->d_name).substr(0, commonPrefix.length());
            if (filePrefix == commonPrefix)
            {
                fileList.push_back(dirp->d_name);
                cout << "[INFO]: Found label " << dirp->d_name << "." << endl;
            }
        }
    }

    cout << "[INFO]: Found " << fileList.size() << " files with the common prefix " << commonPrefix
        << " under " << path << "." << endl;

    closedir(dp);
    return 0;
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
