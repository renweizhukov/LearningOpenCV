/*
 * VocabularyBuilder.cpp
 *
 *  Created on: Jul 10, 2017
 *      Author: renwei
 */

#include "Utility.h"
#include "VocabularyBuilder.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

typedef std::chrono::high_resolution_clock Clock;

VocabularyBuilder::VocabularyBuilder() :
    m_cntBowClusters(0),
    m_surfMinHessian(0)
{
}

VocabularyBuilder::VocabularyBuilder(
    const string& imgBasePath,
    const string& descriptorsFile,
    const string& vocabularyFile) :
    m_imgBasePath(imgBasePath),
    m_descriptorsFile(descriptorsFile),
    m_vocabularyFile(vocabularyFile),
    m_cntBowClusters(1000), // TODO: Expose m_cntBowClusters and m_surfMinHessian as configurable parameters.
    m_surfMinHessian(400)
{

}

VocabularyBuilder::~VocabularyBuilder()
{
}

void VocabularyBuilder::Reset(
    const string& imgBasePath,
    const string& descriptorsFile,
    const string& vocabularyFile)
{
    m_imgBasePath = imgBasePath;
    m_descriptorsFile = descriptorsFile;
    m_vocabularyFile = vocabularyFile;

    m_descriptors.release();
    m_vocabulary.release();
}

void VocabularyBuilder::ComputeDescriptors(OutputArray descriptors)
{
    vector<pair<string, string> > imgWithLabels;
    Utility::GetImagesWithLabels(m_imgBasePath, imgWithLabels);

    FileStorage fs(m_descriptorsFile, FileStorage::WRITE);

    fs << "image_label_list" << "[";
    for (const auto& labelledImg : imgWithLabels)
    {
        // We write the image filename first and then its label.
        fs << labelledImg.second << labelledImg.first;
    }
    fs << "]";  // End of image_label_list
    cout << "[INFO]: Write the filenames of " << imgWithLabels.size() << " images with their labels to file "
        << m_descriptorsFile << "." << endl;

    Ptr<SurfFeatureDetector> detector = SURF::create(m_surfMinHessian);

    cout << "[INFO]: Computing the SURF descriptors of " << imgWithLabels.size() << " images." << endl;
    auto tStart = Clock::now();

    int imgIndex = 0;
    for (const auto& labelledImg : imgWithLabels)
    {
        string imgLabel = labelledImg.first;
        string imgFile = labelledImg.second;
        string imgFullPath = m_imgBasePath + "/" + imgLabel + "/" + imgFile;

        Mat img = imread(imgFullPath);
        vector<KeyPoint> imgKeypoints;
        Mat imgDescriptors;

        detector->detectAndCompute(img, noArray(), imgKeypoints, imgDescriptors);

        // Key names must start with a letter or '_'. Since the image filename may start with a non-letter,
        // e.g., a digit, we don't use the image filename as the key name.
        fs << "descriptors_" + to_string(imgIndex++) << imgDescriptors;
        cout << "[INFO]: Write " << imgDescriptors.rows << " descriptors of image " << imgFile
            << " with label " << imgLabel << " to file " << m_descriptorsFile << "." << endl;

        // A big Mat of descriptors without labels will be the input for building the vocabulary.
        m_descriptors.push_back(imgDescriptors);
    }

    auto tEnd = Clock::now();
    cout << "[INFO]: Get " << m_descriptors.rows << " total descriptors in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " ms." << endl;

    fs.release();

    if (descriptors.needed())
    {
        m_descriptors.copyTo(descriptors);
    }
}

void VocabularyBuilder::BuildVocabulary(OutputArray vocabulary)
{
    cout << "[INFO]: Building the vocabulary." << endl;

    BOWKMeansTrainer bowTrainer(m_cntBowClusters);

    auto tStart = Clock::now();

    bowTrainer.add(m_descriptors);
    m_vocabulary = bowTrainer.cluster();

    auto tEnd = Clock::now();

    cout << "[INFO]: Build the vocabulary in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " ms." << endl;

    FileStorage fs(m_vocabularyFile, FileStorage::WRITE);
    // TODO: Write the count of clusters, the descriptor type (i.e., "SURF"), and the descriptor
    // parameter value (e.g., m_surfMinHessian) to the vocabulary file.
    fs << "vocabulary" << m_vocabulary;

    cout << "[INFO]: Write the vocabulary with " << m_vocabulary.rows << " clusters to file "
        << m_vocabularyFile << "." << endl;

    fs.release();

    if (vocabulary.needed())
    {
        m_vocabulary.copyTo(vocabulary);
    }
}
