/*
 * Utility.h
 *
 *  Created on: Aug 15, 2017
 *      Author: renwei
 */

#ifndef INCLUDES_UTILITY_H_
#define INCLUDES_UTILITY_H_

#include <iostream>
#include <string>
#include <vector>

#include <sys/types.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

class Utility
{
public:
    static int LoadAllImagesInDir(
        const std::string& imgDir,
        std::vector<std::string>& imgFilenameList,
        std::vector<cv::Mat>& imgs);

    static void SeparateDirFromFilename(
        const std::string& fullFilename,
        std::string& dir,
        std::string& filename);

    static std::string CvType2Str(const int type);
};

#endif /* INCLUDES_UTILITY_H_ */
