/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "MapLine.h"    //added for MapLine visualization
#include "Atlas.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<mutex>
#include <unordered_set>


namespace ORB_SLAM3
{

class Tracking;
class Viewer;

class FrameDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
    FrameDrawer(Atlas* pAtlas);

    // Update info from the last processed frame.
    void Update(Tracking *pTracker);

    // Draw last processed frame.
    cv::Mat DrawFrame(float imageScale=1.f);
    cv::Mat DrawRightFrame(float imageScale=1.f);

    bool both;

protected:

    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // Info of the frame to be drawn
    cv::Mat mIm, mImRight;
    int N, NL;
    vector<cv::KeyPoint> mvCurrentKeys,mvCurrentKeysRight;
    std::vector<cv::line_descriptor::KeyLine> mvCurrentKeysLine; //added for line feature
    //cv::Mat mCurrentDescriptors,mCurrentDescriptorsRight;
    vector<bool> mvbMap, mvbVO;
    bool mbOnlyTracking;
    int mnTracked, mnTrackedVO;
    vector<cv::KeyPoint> mvIniKeys;
    std::vector<cv::line_descriptor::KeyLine> mvIniKeysLine; //added for line feature
    vector<int> mvIniMatches;
    int mState;
    std::vector<float> mvCurrentDepth;
    float mThDepth;

    Atlas* mpAtlas;

    std::mutex mMutex;
    vector<pair<cv::Point2f, cv::Point2f> > mvTracks;

    Frame mCurrentFrame;
    vector<MapPoint*> mvpLocalMap;
    std::vector<MapLine*> mvpLocalMapLines; //added for MapLine visualization
    vector<cv::KeyPoint> mvMatchedKeys;
    std::vector<cv::line_descriptor::KeyLine> mvMatchedKeysLine; //added for line feature
    vector<MapPoint*> mvpMatchedMPs;
    std::vector<MapLine*> mvpMatchedMLs; //added for line feature
    vector<cv::KeyPoint> mvOutlierKeys;
    std::vector<cv::line_descriptor::KeyLine> mvOutlierKeysLine; //added for line feature
    vector<MapPoint*> mvpOutlierMPs;
    std::vector<MapLine*> mvpOutlierMLs; //added for line feature

    map<long unsigned int, cv::Point2f> mmProjectPoints;
    std::map<long unsigned int, std::pair<cv::Point2f,cv::Point2f> > mmProjectLines; //added for line feature
    map<long unsigned int, cv::Point2f> mmMatchedInImage;

};

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H
