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

vector<pair<string, string> > FlannBasedSavableMatcher::getTrainedImgFilename2LabelList()
{
    return trainedImgFilename2LabelList;
}

void FlannBasedSavableMatcher::setTrainedImgFilename2LabelList(const vector<pair<string, string> >& imgFilename2LabelList)
{
    trainedImgFilename2LabelList = imgFilename2LabelList;
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
    auto tStart = Clock::now();

    // Read indexParams and searchParams from fs.
    FlannBasedMatcher::read(fn);

    // Read the trained image filenames from fs.
    FileNode imgFilenameListNode = fn["imgFilename2LabelList"];
    if (imgFilenameListNode.type() != FileNode::SEQ)
    {
        cerr << "[ERROR]: the list of trained image filenames is not a sequence." << endl << endl;
        return;
    }

    trainedImgFilename2LabelList.clear();
    for (FileNodeIterator itNode = imgFilenameListNode.begin(); itNode != imgFilenameListNode.end(); ++itNode)
    {
        string imgFilename = string(*itNode++);
        string imgLabel = string(*itNode);
        trainedImgFilename2LabelList.push_back(make_pair(imgFilename, imgLabel));
    }

    // Read the trained descriptors from fs.
    vector<Mat> allTrainedDescriptors;
    auto tDescriptorsStart = Clock::now();
    for (size_t imgIndex = 0; imgIndex < trainedImgFilename2LabelList.size(); ++imgIndex)
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

    auto tEnd = Clock::now();
    cout << "[DEBUG]: FlannBasedSavableMatcher::read in " << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " ms." << endl;
}

void FlannBasedSavableMatcher::write(FileStorage& fs) const
{
    // Write indexParams and searchParams into fs.
    FlannBasedMatcher::write(fs);

    // Write the trained image filenames with their labels into fs.
    fs << "imgFilename2LabelList" << "[";
    for (const auto& img2Label : trainedImgFilename2LabelList)
    {
        fs << img2Label.first << img2Label.second;
    }
    fs << "]";  // End of trainedImgFilename2LabelList

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
