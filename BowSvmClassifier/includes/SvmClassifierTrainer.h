/*
 * SvmClassifierTrainer.h
 *
 *  Created on: Jul 10, 2017
 *      Author: renwei
 */

#ifndef INCLUDES_SVMCLASSIFIERTRAINER_H_
#define INCLUDES_SVMCLASSIFIERTRAINER_H_

#include <iostream>
#include <string>
#include <map>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>

class SvmClassifierTrainer
{
private:

    std::string m_vocabularyFile;
    std::string m_descriptorsFile;
    std::string m_imgBasePath;
    std::string m_matcherPrefix;
    std::string m_classifierFilePrefix;

    int m_surfMinHessian;

    std::map<std::string, cv::Mat> m_img2DescriptorsMap;
    std::map<std::string, cv::Mat> m_label2BowDescriptorsMap;

    SvmClassifierTrainer();

    bool ComputeBowDescriptors();
    void TrainAndSaveSvms();
    void TrainAndSaveFlannMatcher();

public:

    SvmClassifierTrainer(
        const std::string& vocabularyFile,
        const std::string& descriptorsFile,
        const std::string& imgBasePath,
        const std::string& matcherPrefix,
        const std::string& classifierFilePrefix);
    ~SvmClassifierTrainer();

    void Reset(
        const std::string& vocabularyFile,
        const std::string& descriptorsFile,
        const std::string& imgBasePath,
        const std::string& matcherPrefix,
        const std::string& classifierFilePrefix);

    void Train();
};

#endif /* INCLUDES_SVMCLASSIFIERTRAINER_H_ */
