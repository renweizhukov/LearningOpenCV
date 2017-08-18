/*
 * Utility.h
 *
 *  Created on: Jul 10, 2017
 *      Author: renwei
 */

#ifndef INCLUDES_UTILITY_H_
#define INCLUDES_UTILITY_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <sys/types.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>

#include <opencv2/core.hpp>

class Utility
{
private:
    static int GetImageLabels(
        const std::string &basePath,
        std::vector<std::string>& imgLabels);

public:
    static int GetImagesWithLabels(
        const std::string& basePath,
        std::vector<std::pair<std::string, std::string> >& imgsWithLabels);

    static int GetLabelImageMap(
        const std::string& basePath,
        std::map<std::string, std::vector<std::string> >& label2ImgMap);

    static int GetFilesWithCommonPrefix(
        const std::string& path,
        const std::string& commonPrefix,
        std::vector<std::string>& fileList);

    static void SeparateDirFromFilename(
        const std::string& fullFilename,
        std::string& dir,
        std::string& filename);

    static std::string CvType2Str(const int type);
};

#endif /* INCLUDES_UTILITY_H_ */
