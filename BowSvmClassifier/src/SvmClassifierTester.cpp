/*
 * SvmClassifierTester.cpp
 *
 *  Created on: Jul 16, 2017
 *      Author: renwei
 */

#include "Utility.h"
#include "SvmClassifierTester.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

typedef std::chrono::high_resolution_clock Clock;

static void write(
    FileStorage& fs,
    const string&,
    const ClassifierResult& classifierResult)
{
    classifierResult.write(fs);
}

static void read(
    const FileNode& node,
    ClassifierResult& classifierResult,
    const ClassifierResult& defaultValue = ClassifierResult())
{
    if (node.empty())
    {
        classifierResult = defaultValue;
    }
    else
    {
        classifierResult.read(node);
    }
}

SvmClassifierTester::SvmClassifierTester():
    m_surfMinHessian(0)
{
}

SvmClassifierTester::SvmClassifierTester(
    const string& vocabularyFile,
    const string& classifierFilePrefix,
    const string& resultFile) :
    m_surfMinHessian(400),  // TODO: Read the value from the vocabulary file.
    m_vocabularyFile(vocabularyFile),
    m_classifierFilePrefix(classifierFilePrefix),
    m_resultFile(resultFile)
{

}

SvmClassifierTester::~SvmClassifierTester()
{
}

void SvmClassifierTester::Reset(
    const string& vocabularyFile,
    const string& classifierFilePrefix,
    const string& resultFile)
{
    m_vocabularyFile = vocabularyFile;
    m_classifierFilePrefix = classifierFilePrefix;
    m_resultFile = resultFile;

    m_img2BowDescriptorMap.clear();
    m_img2ClassifierResultMap.clear();
    m_class2SvmMap.clear();

    InitBowImgDescriptorExtractor();
    InitSvmClassifiers();
}

bool SvmClassifierTester::InitBowImgDescriptorExtractor()
{
    // Load the vocabulary from the vocabulary file.
    Mat vocabulary;
    FileStorage fsVocabulary(m_vocabularyFile, FileStorage::READ);
    fsVocabulary["vocabulary"] >> vocabulary;
    cout << "[INFO]: Read the vocabulary with " << vocabulary.rows << " clusters from " << m_vocabularyFile
        << "." << endl;
    fsVocabulary.release();

    //cout << "[DEBUG]: vocabulary #rows = " << vocabulary.rows << ", #cols = " << vocabulary.cols << ", type = "
    //    << Utility::CvType2Str(vocabulary.type()) << "." << endl;

    // Create the SURF detector, the Brute-Force DescriptorMatcher, and the BOWImgDescriptorExtractor.
    m_detector = SURF::create(m_surfMinHessian);
    m_descMatcher = DescriptorMatcher::create("BruteForce");
    m_bowExtractor.reset(new BOWImgDescriptorExtractor(m_descMatcher));

    // Set the vocabulary.
    m_bowExtractor->setVocabulary(vocabulary);

    return true;
}

bool SvmClassifierTester::InitSvmClassifiers()
{
    // Get the base path and the common filename prefix of all the classifier files from m_classifierFilePrefix;
    size_t slashPos = m_classifierFilePrefix.find_last_of('/');
    string classifierPath = m_classifierFilePrefix.substr(0, slashPos);
    string classifierCommonPrefix = m_classifierFilePrefix.substr(slashPos + 1);

    // Get all the classifier files under classifierBasePath.
    vector<string> classifierFiles;
    Utility::GetFilesWithCommonPrefix(classifierPath, classifierCommonPrefix, classifierFiles);

    // Initialize one 1-vs-all SVM classifier per each class.
    for (const auto& classifierFile : classifierFiles)
    {
        size_t underscorePos = classifierFile.find('_');
        size_t dotPos = classifierFile.find_last_of('.');
        string className = classifierFile.substr(underscorePos + 1, dotPos - underscorePos - 1);

        string classifierFullFilename = classifierPath + "/" + classifierFile;
        Ptr<SVM> svmClassifier = SVM::load(classifierFullFilename);

        //cout << "[DEBUG]: SVM classifier for class " << className << ": varCount = " << svmClassifier->getVarCount() << "." << endl;

        m_class2SvmMap.insert(make_pair(className, svmClassifier));
    }

    return true;
}

bool SvmClassifierTester::ComputeBowDescriptor(
    const std::string& img2ClassifierResultMapKey,
    const cv::Mat& img)
{
    // Compute the SURF descriptors.
    vector<KeyPoint> surfKeypoints;
    Mat surfDescriptors;

    m_detector->detectAndCompute(img, noArray(), surfKeypoints, surfDescriptors);

    // Compute the BOW descriptor.
    Mat bowDescriptor;
    m_bowExtractor->compute(surfDescriptors, bowDescriptor);

    //cout << "[DEBUG]: BOW descriptor of image " << img2ClassifierResultMapKey << ": #rows = " << bowDescriptor.rows
    //    << ", #cols = " << bowDescriptor.cols << ", type = " << Utility::CvType2Str(bowDescriptor.type()) << "." << endl;

    m_img2BowDescriptorMap.insert(make_pair(img2ClassifierResultMapKey, bowDescriptor));

    return true;
}

bool SvmClassifierTester::EvaluateOneImgInternal(
    const std::string& img2ClassifierResultMapKey,
    const cv::Mat& img)
{
    auto tStart = Clock::now();

    // Compute the BOW descriptor.
    if (!ComputeBowDescriptor(img2ClassifierResultMapKey, img))
    {
        cerr << "[ERROR]: Failed to compute the BOW descriptor for " << img2ClassifierResultMapKey << "." << endl << endl;
        return false;
    }

    auto tEnd = Clock::now();
    cout << "[INFO]: Compute the BOW descriptor of " << img2ClassifierResultMapKey << " in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " milliseconds." << endl;

    tStart = Clock::now();

    // Test each 1-vs-all SVM and select the one with the best score.
    // TODO: Consider a more sophisticated approach for selecting the one with the best score.
    // For details, please refer to https://github.com/royshil/FoodcamClassifier/blob/master/predict_common.cpp
    float minDecisionFuncVal = FLT_MAX;
    string minClass;
    for (const auto& svm : m_class2SvmMap)
    {
        float decisionFuncVal = svm.second->predict(m_img2BowDescriptorMap[img2ClassifierResultMapKey], noArray(), true);
        m_img2ClassifierResultMap[img2ClassifierResultMapKey].class2ScoresMap.insert(make_pair(svm.first, decisionFuncVal));

        if (decisionFuncVal < minDecisionFuncVal)
        {
            minDecisionFuncVal = decisionFuncVal;
            minClass = svm.first;
        }
    }

    tEnd = Clock::now();
    cout << "[INFO]: Test all 1-vs-all classifiers of " << img2ClassifierResultMapKey << " in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " milliseconds." << endl;

    m_img2ClassifierResultMap[img2ClassifierResultMapKey].evaluatedClass = minClass;

    return true;
}

void SvmClassifierTester::WriteResultsToFile()
{
    cout << "===============================================================================================" << endl;
    cout << "[INFO]: Evaluated the classes of " << m_img2ClassifierResultMap.size() << " images." << endl;
    cout << "===============================================================================================" << endl;

    for (auto& imgResult : m_img2ClassifierResultMap)
    {
        cout << "[INFO]: image " << imgResult.first << ": expected class = " << imgResult.second.expectedClass
            << ", evaluated class = " << imgResult.second.evaluatedClass << " with score = "
            << imgResult.second.class2ScoresMap[imgResult.second.evaluatedClass] << "." << endl;

        //cout << "[DEBUG]: image " << imgResult.first << ": scores" << endl;
        //for (const auto& classScore : imgResult.second.class2ScoresMap)
        //{
        //    cout << "[DEBUG]: \t\t class = " << classScore.first << ", score = " << classScore.second << "." << endl;
        //}
    }

    FileStorage fsResult(m_resultFile, FileStorage::WRITE);

    // Write the trained class list to the result file.
    vector<string> trainedClassList;
    for (const auto& classSvm : m_class2SvmMap)
    {
        trainedClassList.push_back(classSvm.first);
    }
    fsResult << "trained_class_list" << trainedClassList;

    // Write the image list to the result file.
    vector<string> imgList;
    for (const auto& imgBowDescriptor : m_img2BowDescriptorMap)
    {
        imgList.push_back(imgBowDescriptor.first);
    }
    fsResult << "image_list" << imgList;

    // Write the classifier result of the image to the result file.
    for (const auto& imgClassifierResult : m_img2ClassifierResultMap)
    {
        // For OpenCV FileStorage, key names may only contain alphanumeric characters [a-zA-Z0-9],
        // '-', '_' and ' '. Unfortunately key names may not contain '.'. Also key names must
        // start with a letter or '_'. Since the image filename may start with a non-letter,
        // e.g., a digit, we have to put it after those prefixes.
        size_t dotPos = imgClassifierResult.first.find_last_of('.');
        string classifierResultFsKey = "classifier_result_" + imgClassifierResult.first.substr(0, dotPos);

        fsResult << classifierResultFsKey << imgClassifierResult.second;
    }

    fsResult.release();
}

void SvmClassifierTester::EvaluateOneImg(
    const string& imgFullFilename,
    const string& expectedClass)
{
    auto tStart = Clock::now();

    m_img2BowDescriptorMap.clear();
    m_img2ClassifierResultMap.clear();

    ClassifierResult result;
    result.expectedClass = expectedClass;

    // Remove the path before the image full filename and use only the image filename as the map key.
    size_t slashPos = imgFullFilename.find_last_of('/');
    string imgFilename = imgFullFilename.substr(slashPos + 1);

    m_img2ClassifierResultMap.insert(make_pair(imgFilename, result));

    Mat img = imread(imgFullFilename);
    if (!EvaluateOneImgInternal(imgFilename, img))
    {
        // If we fail to evaluate the class of the test image, we empty the string evaluatedClass
        // and clear the map class2ScoresMap.
        m_img2ClassifierResultMap[imgFilename].evaluatedClass.clear();
        m_img2ClassifierResultMap[imgFilename].class2ScoresMap.clear();
    }

    auto tEnd = Clock::now();
    cout << "[INFO]: Evaluate the class of image " << imgFullFilename << " in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " milliseconds." << endl;

    // Write the evaluated result (i.e., the evaluated class and scores) of the test image
    // along with its expected class to the result file.
    WriteResultsToFile();
}

void SvmClassifierTester::EvaluateImgs(const string& imgBasePath)
{
    auto tStart = Clock::now();

    m_img2BowDescriptorMap.clear();
    m_img2ClassifierResultMap.clear();

    // Get all the test images under the base path along with their expected classes. Note that
    // the expected class of a test image is denoted by the name of the sub-directory where the
    // test image is located under the base path.
    vector<pair<string, string> > imgWithLabels;
    Utility::GetImagesWithLabels(imgBasePath, imgWithLabels);
    for (const auto& labelledImg : imgWithLabels)
    {
        ClassifierResult result;
        result.expectedClass = labelledImg.first;

        m_img2ClassifierResultMap.insert(make_pair(labelledImg.second, result));
    }

    // Evaluate the classes of the test images one by one.
    for (auto& imgFile : m_img2ClassifierResultMap)
    {
        string imgFullFilename = imgBasePath + "/" + imgFile.second.expectedClass + "/" + imgFile.first;
        Mat img = imread(imgFullFilename);

        if (!EvaluateOneImgInternal(imgFile.first, img))
        {
            // If we fail to evaluate the class of one test image, we empty the string evaluatedClass
            // and clear the map class2ScoresMap.
            imgFile.second.evaluatedClass.clear();
            imgFile.second.class2ScoresMap.clear();
        }
    }

    auto tEnd = Clock::now();
    cout << "[INFO]: Evaluate the classes of " << imgWithLabels.size() << " images in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " milliseconds." << endl;

    // Write the evaluated results (i.e., the evaluated classes and scores) of all the test images
    // along with their expected classes to the result file.
    WriteResultsToFile();
}
