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
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>

class SvmClassifierTrainer
{
private:

    struct ImgDescriptorInfo
    {
        std::string label;
        cv::Mat descriptors;
    };

    std::string m_vocabularyFile;
    std::string m_descriptorsFile;
    std::string m_classifierFilePrefix;

    std::map<std::string, ImgDescriptorInfo> m_img2DescriptorInfoMap;
    std::map<std::string, cv::Mat> m_label2BowDescriptorsMap;

    SvmClassifierTrainer();

    bool ComputeBowDescriptors();
    void TrainAndSaveSvms();

public:

    SvmClassifierTrainer(
        const std::string& vocabularyFile,
        const std::string& descriptorsFile,
        const std::string& classifierFilePrefix);
    ~SvmClassifierTrainer();

    void Reset(
        const std::string& vocabularyFile,
        const std::string& descriptorsFile,
        const std::string& classifierFilePrefix);

    void Train();
};

#endif /* INCLUDES_SVMCLASSIFIERTRAINER_H_ */
