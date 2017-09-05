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
    m_surfMinHessian(0),
    m_knnMatchCandidateCnt(0),
    m_goodMatchPercentThreshold(0.0),
    m_goodMatchCntThreshold(0)
{
}

SvmClassifierTester::SvmClassifierTester(
    const string& vocabularyFile,
    const string& classifierFilePrefix,
    const string& matcherDescriptorsFile,
    const string& resultFile) :
    m_surfMinHessian(400),  // TODO: Read the value from the vocabulary file.
    m_knnMatchCandidateCnt(5),
    m_goodMatchPercentThreshold(7.5),
    m_goodMatchCntThreshold(10),
    m_vocabularyFile(vocabularyFile),
    m_classifierFilePrefix(classifierFilePrefix),
    m_matcherDescriptorsFile(matcherDescriptorsFile),
    m_resultFile(resultFile)
{

}

SvmClassifierTester::~SvmClassifierTester()
{
}

void SvmClassifierTester::Reset(
    const string& vocabularyFile,
    const string& classifierFilePrefix,
    const string& matcherDescriptorsFile,
    const string& resultFile)
{
    m_vocabularyFile = vocabularyFile;
    m_classifierFilePrefix = classifierFilePrefix;
    m_matcherDescriptorsFile = matcherDescriptorsFile;
    m_resultFile = resultFile;

    m_img2BowDescriptorMap.clear();
    m_img2ClassifierResultMap.clear();
    m_class2SvmMap.clear();

    InitBowImgDescriptorExtractor();
    InitSvmClassifiers();
    LoadMatcherDescriptors();
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
    string classifierDir;
    string classifierPrefix;
    Utility::SeparateDirFromFilename(m_classifierFilePrefix, classifierDir, classifierPrefix);

    // Get all the classifier files under classifierDir.
    vector<string> classifierFiles;
    Utility::GetFilesWithCommonPrefix(classifierDir, classifierPrefix, classifierFiles);

    // Initialize one 1-vs-all SVM classifier per each class.
    for (const auto& classifierFile : classifierFiles)
    {
        size_t underscorePos = classifierFile.find('_');
        size_t dotPos = classifierFile.find_last_of('.');
        string className = classifierFile.substr(underscorePos + 1, dotPos - underscorePos - 1);

        string classifierFullFilename = classifierDir + "/" + classifierFile;
        Ptr<SVM> svmClassifier = SVM::load(classifierFullFilename);

        //cout << "[DEBUG]: SVM classifier for class " << className << ": varCount = " << svmClassifier->getVarCount() << "." << endl;

        m_class2SvmMap.insert(make_pair(className, svmClassifier));
    }

    return true;
}

bool SvmClassifierTester::LoadMatcherDescriptors()
{
    // Load the filenames of all the training images.
    FileStorage fsMatcherDescriptors(m_matcherDescriptorsFile, FileStorage::READ);
    FileNode img2LabelListNode = fsMatcherDescriptors["image_label_list"];
    if (img2LabelListNode.type() != FileNode::SEQ)
    {
        cerr << "[ERROR]: The list of image filenames with labels is not a sequence in " << m_matcherDescriptorsFile
            << "." << endl << endl;
        return false;
    }

    vector<pair<string, string> > imgFullFilename2LabelList;
    for (FileNodeIterator itNode = img2LabelListNode.begin(); itNode != img2LabelListNode.end(); ++itNode)
    {
        string imgFullFilename = (string)(*itNode++);
        string imgLabel = (string)(*itNode);
        imgFullFilename2LabelList.push_back(make_pair(imgFullFilename, imgLabel));
    }

    cout << "[INFO]: Read the filenames of " << imgFullFilename2LabelList.size() << " images with their labels from "
        << m_matcherDescriptorsFile << " for the FLANN-based matcher." << endl;

    // Load the descriptors with the labels of the images for the FLANN-based matcher.
    int imgIndex = 0;
    for (const auto& imgFullFilename2Label : imgFullFilename2LabelList)
    {
        string imgLabel = imgFullFilename2Label.second;

        string descriptorsKey = "descriptors_" + to_string(imgIndex++);

        Mat descriptors;
        fsMatcherDescriptors[descriptorsKey] >> descriptors;

        //cout << "[DEBUG]: image " << imgFullFilename << ": Read the label " << descInfo.label << " and "
        //    << descInfo.descriptors.rows << " descriptors from " << m_descriptorsFile << "." << endl;

        m_class2MatcherDescriptorsMap.insert(make_pair(imgLabel, descriptors));
    }

    cout << "[INFO]: Read the labels and descriptors of " << imgFullFilename2LabelList.size() << " images from "
        << m_matcherDescriptorsFile << "." << endl;

    fsMatcherDescriptors.release();

    return true;
}

bool SvmClassifierTester::ComputeSurfAndBowDescriptor(
    const string& img2ClassifierResultMapKey,
    const Mat& img)
{
    // Compute the SURF descriptors.
    vector<KeyPoint> surfKeypoints;
    Mat surfDescriptors;

    m_detector->detectAndCompute(img, noArray(), surfKeypoints, surfDescriptors);
    m_img2SurfDescriptorMap.insert(make_pair(img2ClassifierResultMapKey, surfDescriptors));

    // Compute the BOW descriptor.
    Mat bowDescriptor;
    m_bowExtractor->compute(surfDescriptors, bowDescriptor);

    //cout << "[DEBUG]: BOW descriptor of image " << img2ClassifierResultMapKey << ": #rows = " << bowDescriptor.rows
    //    << ", #cols = " << bowDescriptor.cols << ", type = " << Utility::CvType2Str(bowDescriptor.type()) << "." << endl;

    m_img2BowDescriptorMap.insert(make_pair(img2ClassifierResultMapKey, bowDescriptor));

    return true;
}

pair<string, pair<float, int> > SvmClassifierTester::FlannBasedKnnMatch(
    const string& img2ClassifierResultMapKey,
    const vector<pair<string, float> >& flannMatchCandidates)
{
    string bestMatchClass;
    float bestMatchPercent = 0.0;
    int bestMatchCnt = 0;

    auto itSurfDescriptors = m_img2SurfDescriptorMap.find(img2ClassifierResultMapKey);
    if (itSurfDescriptors == m_img2SurfDescriptorMap.end())
    {
        cerr << "[ERROR]: Can't find the SURF descriptors of " << img2ClassifierResultMapKey << "." << endl << endl;
        return make_pair(bestMatchClass, make_pair(bestMatchPercent, bestMatchCnt));
    }

    Mat surfDescriptors = itSurfDescriptors->second;

    vector<string> candidateClassNames;
    vector<Mat> allCandidateDescriptors;
    for (const auto& candidate : flannMatchCandidates)
    {
        string className = candidate.first;
        candidateClassNames.push_back(className);
        allCandidateDescriptors.push_back(m_class2MatcherDescriptorsMap[className]);
    }

    Ptr<FlannBasedMatcher> flannMatcher = FlannBasedMatcher::create();

    vector<vector<DMatch>> knnMatches;

    auto tStart = Clock::now();
    flannMatcher->knnMatch(surfDescriptors, allCandidateDescriptors, knnMatches, 2);
    auto tEnd = Clock::now();
    cout << "[INFO]: Do the FLANN-based knnMatch of " << img2ClassifierResultMapKey << " in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count() << " ms." << endl;

    vector<int> candidateGoodMatchCnts(flannMatchCandidates.size());
    for (const auto& knnMatchPair: knnMatches)
    {
        if (knnMatchPair.size() > 1 && knnMatchPair[0].distance < 0.8 * knnMatchPair[1].distance)
        {
            ++candidateGoodMatchCnts[knnMatchPair[0].imgIdx];
        }
    }

    vector<pair<float, float> > candidateGoodMatchPercentages(flannMatchCandidates.size());
    for (int canIndex = 0; canIndex < static_cast<int>(flannMatchCandidates.size()); ++canIndex)
    {
        candidateGoodMatchPercentages[canIndex].first = 0.0;
        if (surfDescriptors.rows > 0)
        {
            candidateGoodMatchPercentages[canIndex].first
                = 100.0*candidateGoodMatchCnts[canIndex]/(surfDescriptors.rows);
        }

        candidateGoodMatchPercentages[canIndex].second = 0.0;
        if (allCandidateDescriptors[canIndex].rows > 0)
        {
            candidateGoodMatchPercentages[canIndex].second
                = 100.0*candidateGoodMatchCnts[canIndex]/(allCandidateDescriptors[canIndex].rows);
        }

        m_img2ClassifierResultMap[img2ClassifierResultMapKey].class2MatchPercentsMap.insert(
            make_pair(candidateClassNames[canIndex], candidateGoodMatchPercentages[canIndex]));
        m_img2ClassifierResultMap[img2ClassifierResultMapKey].class2MatchCntMap.insert(
            make_pair(candidateClassNames[canIndex], candidateGoodMatchCnts[canIndex]));

        if (candidateGoodMatchPercentages[canIndex].first > bestMatchPercent)
        {
            bestMatchPercent = candidateGoodMatchPercentages[canIndex].first;
            bestMatchCnt = candidateGoodMatchCnts[canIndex];
            bestMatchClass = candidateClassNames[canIndex];
        }

        if (candidateGoodMatchPercentages[canIndex].second > bestMatchPercent)
        {
            bestMatchPercent = candidateGoodMatchPercentages[canIndex].second;
            bestMatchCnt = candidateGoodMatchCnts[canIndex];
            bestMatchClass = candidateClassNames[canIndex];
        }
    }

    return make_pair(bestMatchClass, make_pair(bestMatchPercent, bestMatchCnt));
}

bool SvmClassifierTester::EvaluateOneImgInternal(
    const string& img2ClassifierResultMapKey,
    const Mat& img)
{
    auto tStart = Clock::now();

    // Compute the BOW descriptor.
    if (!ComputeSurfAndBowDescriptor(img2ClassifierResultMapKey, img))
    {
        cerr << "[ERROR]: Failed to compute the BOW descriptor for " << img2ClassifierResultMapKey << "." << endl << endl;
        return false;
    }

    auto tEnd = Clock::now();
    cout << "[INFO]: Compute the BOW descriptor of " << img2ClassifierResultMapKey << " in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " ms." << endl;

    tStart = Clock::now();

    // Test each 1-vs-all SVM and select a couple of candidates with the best scores.
    // TODO: Consider a more sophisticated approach for selecting the ones with the best scores.
    // For details, please refer to https://github.com/royshil/FoodcamClassifier/blob/master/predict_common.cpp
    vector<pair<string, float>> classDecFuncVals;
    for (const auto& svm : m_class2SvmMap)
    {
        float decisionFuncVal = svm.second->predict(m_img2BowDescriptorMap[img2ClassifierResultMapKey], noArray(), true);
        m_img2ClassifierResultMap[img2ClassifierResultMapKey].class2ScoresMap.insert(make_pair(svm.first, decisionFuncVal));

        classDecFuncVals.push_back(make_pair(svm.first, decisionFuncVal));
    }

    tEnd = Clock::now();
    cout << "[INFO]: Test all 1-vs-all classifiers of " << img2ClassifierResultMapKey << " in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " ms." << endl;

    // Create a min-heap from classDecFuncVals;
    make_heap(classDecFuncVals.begin(), classDecFuncVals.end(), ClassDecFuncComparison(true));
    vector<pair<string, float> > flannMatchCandidates;
    for (int canIndex = 0; canIndex < m_knnMatchCandidateCnt; ++canIndex)
    {
        flannMatchCandidates.push_back(classDecFuncVals.front());
        pop_heap(classDecFuncVals.begin(), classDecFuncVals.end(), ClassDecFuncComparison(true));
        classDecFuncVals.pop_back();
    }

    // Do the FLANN-based matching for the candidates. If the maximum percentage of the good matches exceeds a certain
    // threshold m_goodMatchPercentThreshold, then evaluate the class as the one with the maximum percentage; otherwise
    // evaluate the class as "unknown".
    pair<string, pair<float, int> > bestMatch = FlannBasedKnnMatch(img2ClassifierResultMapKey, flannMatchCandidates);

    if (bestMatch.second.first >= m_goodMatchPercentThreshold)
    {
        if (bestMatch.second.second >= m_goodMatchCntThreshold)
        {
            cout << "[INFO]: The maximum match percentage " << bestMatch.second.first << "% of " << img2ClassifierResultMapKey
                << " is above the threshold " << m_goodMatchPercentThreshold << "%, so evaluate the class as "
                << bestMatch.first << "." << endl;
        }
        else
        {
            bestMatch.first = "unknown";
            cout << "[INFO]: Although the maximum match percentage " << bestMatch.second.first << "% of " << img2ClassifierResultMapKey
                << " is above the threshold " << m_goodMatchPercentThreshold << "%, the maximum match count " << bestMatch.second.second
                << " is below the threshold " << m_goodMatchCntThreshold << ", so evaluate the class as unknown." << endl;
        }
    }
    else
    {
        bestMatch.first = "unknown";
        cout << "[INFO]: The maximum match percentage " << bestMatch.second.first << "% of " << img2ClassifierResultMapKey
            << " is below the threshold " << m_goodMatchPercentThreshold << "%, so evaluate the class as unknown." << endl;
    }

    m_img2ClassifierResultMap[img2ClassifierResultMapKey].evaluatedClass = bestMatch.first;

    return true;
}

void SvmClassifierTester::WriteResultsToFile()
{
    cout << "===============================================================================================" << endl;
    cout << "[INFO]: Complete results of " << m_img2ClassifierResultMap.size() << " images:" << endl;
    cout << "===============================================================================================" << endl;

    map<string, ClassifierResult> img2ClassifierResultErrors;
    for (auto& imgResult : m_img2ClassifierResultMap)
    {
        if (imgResult.second.evaluatedClass != imgResult.second.expectedClass)
        {
            img2ClassifierResultErrors.insert(imgResult);
        }

        if (imgResult.second.evaluatedClass != "unknown")
        {
            cout << "[INFO]: " << imgResult.first << ": expected class = " << imgResult.second.expectedClass
                << ", evaluated class = " << imgResult.second.evaluatedClass << " with score = "
                << imgResult.second.class2ScoresMap[imgResult.second.evaluatedClass] << ", matchQueryPercent = "
                << imgResult.second.class2MatchPercentsMap[imgResult.second.evaluatedClass].first << "%, matchTestPercent = "
                << imgResult.second.class2MatchPercentsMap[imgResult.second.evaluatedClass].second << "%, and matchCnt = "
                << imgResult.second.class2MatchCntMap[imgResult.second.evaluatedClass] << "." << endl;
        }
        else
        {
            cout << "[INFO]: " << imgResult.first << ": expected class = " << imgResult.second.expectedClass
                << ", evaluated class = unknown." << endl;
}
    }

    cout << "===============================================================================================" << endl;
    cout << "[INFO]: Error rate = " << 100.0*img2ClassifierResultErrors.size()/m_img2ClassifierResultMap.size() << "%." << endl;
    cout << "[INFO]: " << img2ClassifierResultErrors.size() << " error images:" << endl;
    cout << "===============================================================================================" << endl;

    for (const auto& imgResult : img2ClassifierResultErrors)
    {
        cout << "[INFO]: " << imgResult.first << ": expected class = " << imgResult.second.expectedClass
                << ", evaluated class = " << imgResult.second.evaluatedClass << "." << endl;
    }

    FileStorage fsResult(m_resultFile, FileStorage::WRITE);

    // Write the trained class list to the result file.
    vector<string> trainedClassList;
    for (const auto& classSvm : m_class2SvmMap)
    {
        trainedClassList.push_back(classSvm.first);
    }
    fsResult << "trained_class_list" << trainedClassList;

    // Write the evaluated classes of the test images to the result file.
    fsResult << "evaluated_class_list" << "[";
    for (const auto& imgResult : m_img2ClassifierResultMap)
    {
        fsResult << imgResult.first << imgResult.second.evaluatedClass;
    }
    fsResult << "]";    // End of evaluated_class_list

    // Write the detailed classifier result of the image to the result file.
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
        m_img2ClassifierResultMap[imgFilename].class2MatchPercentsMap.clear();
        m_img2ClassifierResultMap[imgFilename].class2MatchCntMap.clear();
    }

    auto tEnd = Clock::now();
    cout << "[INFO]: Evaluated the class of image " << imgFullFilename << " in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " ms." << endl;

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

    // Evaluate the classes of the test images one by one.
    for (const auto& labelledImg : imgWithLabels)
    {
        string imgLabel = labelledImg.first;
        string imgFilename = labelledImg.second;
        string img2ClassifierResultMapKey = imgLabel + "_" + imgFilename;

        ClassifierResult result;
        result.expectedClass = imgLabel;

        m_img2ClassifierResultMap.insert(make_pair(img2ClassifierResultMapKey, result));

        string imgFullFilename = imgBasePath + "/" + imgLabel + "/" + imgFilename;
        Mat img = imread(imgFullFilename);

        if (!EvaluateOneImgInternal(img2ClassifierResultMapKey, img))
        {
            // If we fail to evaluate the class of one test image, we empty the string evaluatedClass
            // and clear the map class2ScoresMap.
            m_img2ClassifierResultMap[img2ClassifierResultMapKey].evaluatedClass.clear();
            m_img2ClassifierResultMap[img2ClassifierResultMapKey].class2ScoresMap.clear();
            m_img2ClassifierResultMap[img2ClassifierResultMapKey].class2MatchPercentsMap.clear();
            m_img2ClassifierResultMap[img2ClassifierResultMapKey].class2MatchCntMap.clear();
        }
    }

    auto tEnd = Clock::now();
    cout << "[INFO]: Evaluated the classes of " << imgWithLabels.size() << " images in "
        << chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count()
        << " ms." << endl;

    // Write the evaluated results (i.e., the evaluated classes and scores) of all the test images
    // along with their expected classes to the result file.
    WriteResultsToFile();
}
