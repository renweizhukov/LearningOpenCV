/*
 * FlannBasedSavableMatcher.cpp
 *
 *  Created on: Aug 15, 2017
 *      Author: renwei
 */

#include <iostream>
#include <string>
#include <chrono>

#include "FlannBasedSavableMatcher.h"

using namespace cv;
using namespace std;

typedef std::chrono::high_resolution_clock Clock;

FlannBasedSavableMatcher::FlannBasedSavableMatcher() :
    flannIndexFileDir("./")
{

}

FlannBasedSavableMatcher::~FlannBasedSavableMatcher()
{

}

Ptr<FlannBasedSavableMatcher> FlannBasedSavableMatcher::create()
{
    return makePtr<FlannBasedSavableMatcher>();
}

vector<string> FlannBasedSavableMatcher::getTrainedImgFilenameList()
{
    return trainedImgFilenameList;
}

void FlannBasedSavableMatcher::setTrainedImgFilenameList(const vector<string>& imgFilenameList)
{
    trainedImgFilenameList = imgFilenameList;
}

void FlannBasedSavableMatcher::setFlannIndexFileDir(const string& dir)
{
    flannIndexFileDir = dir;
}

void FlannBasedSavableMatcher::setFlannIndexFilename(const string& filename)
{
    flannIndexFilename = filename;
}

void FlannBasedSavableMatcher::read(const FileNode& fn)
{
    // Read indexParams and searchParams from fs.
    FlannBasedMatcher::read(fn);

    // Read the trained image filenames from fs.
    FileNode imgFilenameListNode = fn["imgFilenameList"];
    if (imgFilenameListNode.type() != FileNode::SEQ)
    {
        cerr << "[ERROR]: the list of trained image filenames is not a sequence." << endl << endl;
        return;
    }

    trainedImgFilenameList.clear();
    for (FileNodeIterator itNode = imgFilenameListNode.begin(); itNode != imgFilenameListNode.end(); ++itNode)
    {
        trainedImgFilenameList.push_back(string(*itNode));
    }

    // Read the trained descriptors from fs.
    vector<Mat> allTrainedDescriptors;
    auto tDescriptorsStart = Clock::now();
    for (size_t imgIndex = 0; imgIndex < trainedImgFilenameList.size(); ++imgIndex)
    {
        string descriptorsKey("descriptors_" + to_string(imgIndex));
        Mat descriptors;

        fn[descriptorsKey] >> descriptors;
        allTrainedDescriptors.push_back(descriptors);
    }
    auto tDescriptorsEnd = Clock::now();
    cout << "[DEBUG]: Loaded the descriptors in " << chrono::duration_cast<chrono::milliseconds>(tDescriptorsEnd - tDescriptorsStart).count()
        << " ms." << endl;

    // Add the trained descriptors to the matcher and update all the related descriptors.
    FlannBasedMatcher::add(allTrainedDescriptors);
    if (!utrainDescCollection.empty())
    {
        CV_Assert(trainDescCollection.size() == 0);
        for (size_t i = 0; i < utrainDescCollection.size(); ++i)
        {
            trainDescCollection.push_back(utrainDescCollection[i].getMat(ACCESS_READ));
        }
    }
    mergedDescriptors.set(trainDescCollection);

    // Read flannIndex from "flannIndexFileDir + flannIndexFilename".
    fn["flannIndexFilename"] >> flannIndexFilename;

    cout << "[DEBUG]: Reading the FLANN index from " << flannIndexFileDir + flannIndexFilename << "." << endl;

    flannIndex.release();
    flannIndex = makePtr<flann::Index>();
    auto tFlannIndexStart = Clock::now();
    flannIndex->load(mergedDescriptors.getDescriptors(), flannIndexFileDir + flannIndexFilename);
    auto tFlannIndexEnd = Clock::now();
    cout << "[DEBUG]: Loaded the flannIndex in " << chrono::duration_cast<chrono::milliseconds>(tFlannIndexEnd - tFlannIndexStart).count()
        << " ms." << endl;
}

void FlannBasedSavableMatcher::write(FileStorage& fs) const
{
    // Write indexParams and searchParams into fs.
    FlannBasedMatcher::write(fs);

    // Write the trained image filenames into fs.
    fs << "imgFilenameList" << trainedImgFilenameList;

    // Write the trained descriptors into fs.
    for (size_t imgIndex = 0; imgIndex < trainDescCollection.size(); ++imgIndex)
    {
        fs << string("descriptors_" + to_string(imgIndex)) << trainDescCollection[imgIndex];
    }

    if (!flannIndexFilename.empty())
    {
        // Write flannIndexFile into fs. Since we assume that flannIndexFile is always in the same
        // directory as fs, we don't write flannIndexFileDir into fs.
        fs << "flannIndexFilename" << flannIndexFilename;

        // Write flannIndex into "flannIndexFileDir + flannIndexFilename". Note that flann::Index::save()
        // writes the index data in a raw format and doesn't support serialization, so we have to write
        // the index data into a separate file.
        flannIndex->save(flannIndexFileDir + flannIndexFilename);
    }
    else
    {
        cerr << "[ERROR]: flannIndexFilename is empty so flannIndex is not saved." << endl << endl;
    }
}
