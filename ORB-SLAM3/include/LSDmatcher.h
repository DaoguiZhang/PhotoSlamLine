/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/
/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORB_SLAM2_LSDMATCHER_H
#define ORB_SLAM2_LSDMATCHER_H

#include <opencv2/line_descriptor/descriptor.hpp>
#include "MapLine.h"
#include "KeyFrame.h"
#include "Frame.h"

namespace ORB_SLAM3
{
    class KeyFrame;
    class Frame;
    class MapLine;

    class LSDmatcher
    {
    public:
        static const int TH_HIGH, TH_LOW;

        LSDmatcher(float nnratio=0.6, bool checkOri=true, float ratio=0.85f, float ransac_thresh=3.0f, float maxAngleDiff=30.0f, float maxLengthRatio=2.0f): mfNNratio(nnratio), mbCheckOrientation(checkOri), mratio_thresh(ratio), mransac_threshold(ransac_thresh),
          mmax_angle_diff(maxAngleDiff), mmax_length_ratio(maxLengthRatio) {}
        
        ~LSDmatcher(){}

        // 匹配 keylines + descriptors
        void match(const std::vector<cv::line_descriptor::KeyLine>& keylines1, const cv::Mat& desc1,
               const std::vector<cv::line_descriptor::KeyLine>& keylines2, const cv::Mat& desc2,
               std::vector<cv::DMatch>& good_matches); 

        int SearchByDescriptor(KeyFrame* pKF, Frame &currentF, std::vector<MapLine*> &vpMapLineMatches);
        int SearchByDescriptor(KeyFrame* pKF, KeyFrame *pKF2, std::vector<MapLine*> &vpMapLineMatches);
        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool Mono);
        //int MatchLinesByProjection(Frame &currentFrame, const Frame &lastFrame, const float threshold, const bool isMono);
        int SearchByProjectionNew(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool isMono);
        void DebugSearchByProjectionNew(Frame &CurrentFrame,const Frame &LastFrame, const std::string &windowName);
        void DebugDrawLineMatches(const Frame &lastFrame, const Frame &currentFrame);
        void DebugDrawLineMatchesFrame(Frame &CurrentFrame, std::string &windowName);
        void DebugSearchByProjectionLinesMatch(Frame &F, std::vector<MapLine*> &vpMapLines,const std::string &windowName);
        void DebugLineProjectedKeyFrame(KeyFrame* pKF, std::vector<MapLine*> &vpMapLines, const std::string & winName);
        void DebugLineMatchesTwoFrames(KeyFrame* pKF1,KeyFrame* pKF2,const std::vector<std::pair<MapLine*, MapLine*>> &vpMatchedLines,const std::string &winName);
        void DebugDrawProjectedLineFrame(Frame &CurrentFrame, std::string &windowName);
        int SearchByProjection(KeyFrame* pKF,Frame &currentF, std::vector<MapLine*> &vpMapLineMatches);
        void DebugDrawLineMatchesKeyFrame(KeyFrame* pKF, const Frame &currentFrame);
        int SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, const float th=3);
        void DebugLineMatchSearchbyProjection(Frame &F, const std::vector<MapLine *> &vpMapLines, const float th);
        void DebugLineProjectionNew(Frame &F, const std::vector<MapLine*> &vpMapLines, const std::string &winName = "Line Projection Debug");
        int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapLine*> &vpLines, std::vector<MapLine*> &vpMatched, int th);
        int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapLine *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);
        int SerachForInitialize(Frame &InitialFrame, Frame &CurrentFrame, std::vector<std::pair<int,int>> &LineMatches);
        int SerachForInitializeCV(Frame &InitialFrame, Frame &CurrentFrame, std::vector<std::pair<int, int>> &LineMatches); //use OpenCV functions
        int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<std::pair<int, int>> &vMatchedPairs);
        void SearchForTriangulationLine(KeyFrame *pKF1, KeyFrame *pKF2,vector<pair<int,int>> &vMatchedIdx);
        int SearchForTriangulationFused(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<std::pair<int, int>> &vMatchedPairs);

        // Project MapLines into KeyFrame and search for duplicated MapLines
        int FuseOld(KeyFrame* pKF, const std::vector<MapLine *> &vpMapLines, const float th=3.0);
        int Fuse(KeyFrame* pKF, const std::vector<MapLine *> &vpMapLines, const float th=3.0);

        int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapLine*> &vpLines, float th, std::vector<MapLine *> &vpReplaceLine);

        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        //Draw matches for visualization
        static cv::Mat DrawLineMatches(const cv::Mat &img1, const std::vector<cv::line_descriptor::KeyLine> &keylines1,
                                       const cv::Mat &img2, const std::vector<cv::line_descriptor::KeyLine> &keylines2, std::vector<cv::DMatch> &nmatches);

    protected:
        float RadiusByViewingCos(const float &viewCos);
        float mfNNratio;
        bool mbCheckOrientation;

        float mratio_thresh;
        float mransac_threshold;
        float mmax_angle_diff;     // 最大角度差
        float mmax_length_ratio;   // 最大长度比

        static bool sortByQueryIdx(const std::vector<cv::DMatch>& a, const std::vector<cv::DMatch>& b) {
            if (a.empty() || b.empty()) return false;
            return a[0].queryIdx < b[0].queryIdx;
        }
    };
}


#endif //ORB_SLAM2_LSDMATCHER_H
