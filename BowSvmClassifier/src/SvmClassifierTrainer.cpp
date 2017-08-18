/*
 * SvmClassifierTrainer.cpp
 *
 *  Created on: Jul 14, 2017
 *      Author: renwei
 */

#include "Utility.h"
#include "SvmClassifierTrainer.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

typedef std::chrono::high_resolution_clock Clock;

SvmClassifierTrainer::SvmClassifierTrainer()
{
}

SvmClassifierTrainer::SvmClassifierTrainer(
    const string& vocabularyFile,
    const string& descriptorsFile,
    const string& classifierFilePrefix) :
    m_vocabularyFile(vocabularyFile),
    m_descriptorsFile(descriptorsFile),
    m_classifierFilePrefix(classifierFilePrefix)
{

}

SvmClassifierTrainer::~SvmClassifierTrainer()
{
}

void SvmClassifierTrainer::Reset(
    const std::string& vocabularyFile,
    const std::string& descriptorsFile,
    const std::string& classifierFilePrefix)
{
    m_vocabularyFile = vocabularyFile;
    m_descriptorsFile = descriptorsFile;
    m_classifierFilePrefix = classifierFilePrefix;

    m_img2DescriptorInfoMap.clear();
    m_label2BowDescriptorsMap.clear();
}

bool SvmClassifierTrainer::ComputeBowDescriptors()
{
    // Load the vocabulary from the vocabulary file.
    Mat vocabulary;
    FileStorage fsVocabulary(m_vocabularyFile, FileStorage::READ);
    fsVocabulary["vocabulary"] >> vocabulary;
    cout << "[INFO]: Read the vocabulary with " << vocabulary.rows << " clusters from " << m_vocabularyFile
        << "." << endl;
    fsVocabulary.release();

    // Load the filenames of all the training images.
    FileStorage fsDescriptors(m_descriptorsFile, FileStorage::READ);
    FileNode imgFileListNode = fsDescriptors["imagefile_list"];
    if (imgFileListNode.type() != FileNode::SEQ)
    {
        cerr << "[ERROR]: The list of image filenames is not a sequence in " << m_descriptorsFile
            << "." << endl << endl;
        return false;
    }

    vector<string> imgFullFilenames;
    for (FileNodeIterator itNode = imgFileListNode.begin(); itNode != imgFileListNode.end(); ++itNode)
    {
        imgFullFilenames.push_back((string)(*itNode));
    }

    cout << "[INFO]: Read the filenames of " << imgFullFilenames.size() << " images from "
        << m_descriptorsFile << "." << endl;

    // Load the descriptors with the labels of all the training images.
    // TODO: Define a struct with the fields name, label, descriptors, and its write function so that
    // each struct corresponding to one image can be directly read from the file via FileStorage.
    for (const string& imgFullFilename : imgFullFilenames)
    {
        size_t dotPos = imgFullFilename.find_last_of('.');
        string imgFilename = imgFullFilename.substr(0, dotPos);

        // Get the label key and the descriptors key from the image filename.
        string labelKey = "label_" + imgFilename;
        string descriptorsKey = "descriptors_" + imgFilename;

        ImgDescriptorInfo descInfo;
        fsDescriptors[labelKey] >> descInfo.label;
        fsDescriptors[descriptorsKey] >> descInfo.descriptors;

        //cout << "[DEBUG]: image " << imgFullFilename << ": Read the label " << descInfo.label << " and "
        //    << descInfo.descriptors.rows << " descriptors from " << m_descriptorsFile << "." << endl;

        m_img2DescriptorInfoMap.insert(make_pair(imgFullFilename, descInfo));
    }

    cout << "[INFO]: Read the labels and descriptors of " << imgFullFilenames.size() << " images from "
        << m_descriptorsFile << "." << endl;

    fsDescriptors.release();

    // Create the DescriptorMatcher which is required for creating the BOWImgDescriptorMatcher.
    Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create("BruteForce");

    // Create the BOWImgDescriptorMatcher.
    BOWImgDescriptorExtractor bowExtractor(descMatcher);

    // Set the vocabulary of the BOWImgDescriptorMatcher.
    bowExtractor.setVocabulary(vocabulary);

    auto tStart = Clock::now();

    // Compute the Bag-of-Words (BOW) descriptors from the original image descriptors.
    // A BOW descriptor, a.k.a. a presence vector, is a normalized histogram of vocabulary words
    // encountered in the image. Note that the BOW descriptors are stored in a map with
    // the image label as the key.
    for (const string& imgFullFilename : imgFullFilenames)
    {
        Mat bowDescriptor;
        bowExtractor.compute(m_img2DescriptorInfoMap[imgFullFilename].descriptors, bowDescriptor);

        //cout << "[DEBUG]: image "  << imgFullFilename << ": Compute the BOW descriptor with "
        //    << bowDescriptor.rows << " rows and " << bowDescriptor.cols << " columns." << endl;

        string imgLabel = m_img2DescriptorInfoMap[imgFullFilename].label;
        auto itMap = m_label2BowDescriptorsMap.find(imgLabel);
        if (itMap == m_label2BowDescriptorsMap.end())
        {
            m_label2BowDescriptorsMap.insert(make_pair(m_img2DescriptorInfoMap[imgFullFilename].label, bowDescriptor));
        }
        else
        {
            itMap->second.push_back(bowDescriptor);
        }

        //cout << "[DEBUG]: image "  << imgFullFilename << ": Add the BOW descriptor to class " << imgLabel
        //    << "." << endl;
    }

    auto tEnd = Clock::now();

    cout << "[INFO]: Compute the BOW descriptors of " << imgFullFilenames.size() << " images in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " ms." << endl;

    return true;
}

// Train and save the 1-vs-all SVMs for each class. Note that one class corresponds to one image label.
void SvmClassifierTrainer::TrainAndSaveSvms()
{
    auto tStart = Clock::now();

    for (const auto& labelledBowDescriptors : m_label2BowDescriptorsMap)
    {
        string className = labelledBowDescriptors.first;
        Mat classBowDescriptors = labelledBowDescriptors.second;

        // First we add the BOW descriptors of the current class to the training BOW descriptors and
        // an all-one vector to the training label vector.
        Mat trainingBowDescriptors = classBowDescriptors;
        Mat trainingLabels = Mat::ones(classBowDescriptors.rows, 1, CV_32S);

        // Then we add the BOW descriptors of the remaining classes to the training BOW descriptors
        // and an all-zero vector to the training label vector.
        for (const auto& nonClassLabelledBowDescriptors : m_label2BowDescriptorsMap)
        {
            if (nonClassLabelledBowDescriptors.first != className)
            {
                Mat nonClassBowDescriptors = nonClassLabelledBowDescriptors.second;
                trainingBowDescriptors.push_back(nonClassBowDescriptors);
                trainingLabels.push_back(Mat::zeros(nonClassBowDescriptors.rows, 1, CV_32S));
            }
        }

        if (trainingBowDescriptors.rows == 0)
        {
            cout << "[WARNING]: Class " << className << " has ZERO BOW descriptor!" << endl << endl;
            continue;
        }

        //cout << "[DEBUG]: BOW descriptor element type of class " << className << " = "
        //    << Utility::CvType2Str(trainingBowDescriptors.type()) << "." << endl;

        trainingBowDescriptors.convertTo(trainingBowDescriptors, CV_32F);
        //cout << "[DEBUG]: Covert the element type of the BOW descriptor of class " << className
        //    << " to CV_32F in case that it is not." << endl;

        // Create the SVM classifier and train it using trainingBowDescriptors.
        Ptr<SVM> classifier = SVM::create();
        classifier->train(trainingBowDescriptors, ROW_SAMPLE, trainingLabels);
        cout << "[INFO]: Train the classifier of class " << className << "." << endl;

        // Save the SVM classifier to a file.
        string classifierFilename = m_classifierFilePrefix + "_" + className + ".yml";
        classifier->save(classifierFilename);
    }

    auto tEnd = Clock::now();

    cout << "[INFO]: Train the 1-vs-all SVM classifiers of " << m_label2BowDescriptorsMap.size() << " classes in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " ms." << endl;
}

void SvmClassifierTrainer::Train()
{
    if(ComputeBowDescriptors())
    {
        TrainAndSaveSvms();
    }
}
