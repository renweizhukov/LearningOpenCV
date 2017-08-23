/*
 * VocabularyBuilder.h
 *
 *  Created on: Jul 10, 2017
 *      Author: renwei
 */

#ifndef INCLUDES_VOCABULARYBUILDER_H_
#define INCLUDES_VOCABULARYBUILDER_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>

class VocabularyBuilder
{
private:

    std::string m_imgBasePath;
    std::string m_descriptorsFile;
    std::string m_vocabularyFile;

    int m_cntBowClusters;
    int m_surfMinHessian;
    cv::Mat m_descriptors;
    cv::Mat m_vocabulary;

    VocabularyBuilder();

public:

    VocabularyBuilder(
        const std::string& imgBasePath,
        const std::string& descriptorsFile,
        const std::string& vocabularyFile);
    ~VocabularyBuilder();

    void Reset(
        const std::string& imgBasePath,
        const std::string& descriptorsFile,
        const std::string& vocabularyFile);

    void ComputeDescriptors(cv::OutputArray descriptors);

    void BuildVocabulary(cv::OutputArray vocabulary);
};

#endif /* INCLUDES_VOCABULARYBUILDER_H_ */
