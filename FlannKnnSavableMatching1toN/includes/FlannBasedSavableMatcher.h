/*
 * FlannBasedSavableMatcher.h
 *
 *  Created on: Aug 15, 2017
 *      Author: renwei
 */

#ifndef INCLUDES_FLANNBASEDSAVABLEMATCHER_H_
#define INCLUDES_FLANNBASEDSAVABLEMATCHER_H_

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace cv
{

class FlannBasedSavableMatcher : public FlannBasedMatcher
{
private:
    std::vector<std::string> trainedImgFilenameList;
    std::string flannIndexFileDir;
    std::string flannIndexFilename;

public:
    FlannBasedSavableMatcher();
    virtual ~FlannBasedSavableMatcher();

    std::vector<std::string> getTrainedImgFilenameList();
    void setTrainedImgFilenameList(const std::vector<std::string>& imgFilenameList);
    void setFlannIndexFileDir(const std::string& dir);
    void setFlannIndexFilename(const std::string& filename);

    virtual void read(const FileNode& fn);
    virtual void write(FileStorage& fs) const;

    static Ptr<FlannBasedSavableMatcher> create();
};

}

#endif /* INCLUDES_FLANNBASEDSAVABLEMATCHER_H_ */
