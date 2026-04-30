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

#ifndef LSDEXTRACTOR_H
#define LSDEXTRACTOR_H

#include <opencv2/line_descriptor/descriptor.hpp>
#include <vector>
#include <list>
#include <opencv2/opencv.hpp>


namespace ORB_SLAM3
{

// class ExtractorNode
// {
// public:
//     ExtractorNode():bNoMore(false){}

//     void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

//     std::vector<cv::KeyPoint> vKeys;
//     cv::Point2i UL, UR, BL, BR;
//     std::list<ExtractorNode>::iterator lit;
//     bool bNoMore;
// };

class LSDextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    //LSDextractor(int nfeatures, float scaleFactor, int nlevels,
    //             int iniThFAST, int minThFAST);

    LSDextractor(float minLen = 15.0f, int levels = 1)
        : minLineLength(minLen), nlevels(levels)
    {
       mGridSize = 50;
        mvImagePyramid.resize(levels);
        mvInvScaleFactor = std::vector<float>(levels);
        for (int i = 0; i < levels; ++i) {
            mvInvScaleFactor[i] = 1.0f / std::pow(1.2f, i);
        }
        lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
        lld = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
        
    }

    ~LSDextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::line_descriptor::KeyLine>& _keyplines,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    void SetMinLineLength(float length) {
        minLineLength = length;
    }
    void SetGridSize(int size) {
        mGridSize = size;
    }

    float GetMinLineLength() const {
        return minLineLength;
    }

    float GetGridSize() const {
        return mGridSize;
    }

    // int inline GetLevels(){
    //     return nlevels;}
    // float inline GetScaleFactor(){
    //     return scaleFactor;}
    // std::vector<float> inline GetScaleFactors(){
    //     return mvScaleFactor;
    // }
    // std::vector<float> inline GetInverseScaleFactors(){
    //     return mvInvScaleFactor;
    // }
    // std::vector<float> inline GetScaleSigmaSquares(){
    //     return mvLevelSigma2;
    // }
    // std::vector<float> inline GetInverseScaleSigmaSquares(){
    //     return mvInvLevelSigma2;
    // }

    std::vector<cv::Mat> mvImagePyramid;    //to do next...

    //implementation of detectAndCompute
    void detectAndCompute(const cv::Mat& img, std::vector<cv::line_descriptor::KeyLine>& keylines, cv::Mat& descriptors);

    //filtered: 1: 引入局部密度抑制（Grid-based Suppression） 2:改进长度过滤逻辑 3:线段合并（可选，计算量稍大）
    void detectAndComputeFiltered(const cv::Mat& img, std::vector<cv::line_descriptor::KeyLine>& keylines, cv::Mat& descriptors);

private:

    void ComputeImagePyramid(const cv::Mat& image);
    
    int nlevels;
    // float scaleFactor = 1.2f;
    std::vector<float> mvInvScaleFactor;
    cv::Ptr<cv::LineSegmentDetector> lsd;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> lld;
    float minLineLength;
    int mGridSize = 50; // 网格大小可以根据分辨率调整

    //void ComputePyramid(cv::Mat image);
    //void ComputeKeyLineOctTree(std::vector<std::vector<cv::line_descriptor::KeyLine> >& allKeylines);    
    //std::vector<cv::line_descriptor::KeyLine> DistributeOctTree(const std::vector<cv::line_descriptor::KeyLine>& vToDistributeKeys, const int &minX,
    //                                       const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);
    //void ComputeKeyLinesOld(std::vector<std::vector<cv::line_descriptor::KeyLine> >& allKeylines);
    //std::vector<cv::line_descriptor::KeyLine> pattern;
    //int nfeatures;
    //double scaleFactor;
    //int nlevels;
    //int iniThFAST;
    //int minThFAST;
    //std::vector<int> mnFeaturesPerLevel;
    //std::vector<int> umax;
    //std::vector<float> mvScaleFactor;
    //std::vector<float> mvInvScaleFactor;    
    //std::vector<float> mvLevelSigma2;
    //std::vector<float> mvInvLevelSigma2;
};

} //namespace ORB_SLAM

#endif

