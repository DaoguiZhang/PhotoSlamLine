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


#ifndef MAPLINE_H
#define MAPLINE_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "Converter.h"
#include "SerializationUtils.h"

#include <opencv2/core/core.hpp>
#include <mutex>
#include <utility>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>

namespace ORB_SLAM3
{

class KeyFrame;
class Map;
class Frame;

class MapLine
{


    friend class boost::serialization::access;  //to do serialization next
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & mnId;
        ar & mnFirstKFid;
        ar & mnFirstFrame;
        ar & nObs;
        // Variables used by the tracking
        //ar & mTrackProjX;
        //ar & mTrackProjY;
        //ar & mTrackDepth;
        //ar & mTrackDepthR;
        //ar & mTrackProjXR;
        //ar & mTrackProjYR;
        //ar & mbTrackInView;
        //ar & mbTrackInViewR;
        //ar & mnTrackScaleLevel;
        //ar & mnTrackScaleLevelR;
        //ar & mTrackViewCos;
        //ar & mTrackViewCosR;
        //ar & mnTrackReferenceForFrame;
        //ar & mnLastFrameSeen;
        // Variables used by local mapping
        //ar & mnBALocalForKF;
        //ar & mnFuseCandidateForKF;
        // Variables used by loop closing and merging
        //ar & mnLoopPointForKF;
        //ar & mnCorrectedByKF;
        //ar & mnCorrectedReference;
        //serializeMatrix(ar,mPosGBA,version);
        //ar & mnBAGlobalForKF;
        //ar & mnBALocalForMerge;
        //serializeMatrix(ar,mPosMerge,version);
        //serializeMatrix(ar,mNormalVectorMerge,version);

        // Protected variables
        ar & boost::serialization::make_array(mLineWorldPos.data(), mLineWorldPos.size());
        ar & boost::serialization::make_array(mLsWorldPos.data(), mLsWorldPos.size());
        ar & boost::serialization::make_array(mLeWorldPos.data(), mLeWorldPos.size());
        ar & boost::serialization::make_array(mLineNormalVector.data(), mLineNormalVector.size());
        //ar & BOOST_SERIALIZATION_NVP(mBackupObservationsId);
        //ar & mObservations;
        ar & mLineBackupObservationsId1;
        ar & mLineBackupObservationsId2;
        //serializeMatrix(ar,mDescriptor,version);
        serializeMatrix(ar,mLineDescriptor,version);
        ar & mBackupRefKFId;
        ar & mnLineVisible;
        ar & mnLineFound;
        ar & mbLineBad;
        ar & mLineBackupReplacedId;
        ar & mfMinDistance;
        ar & mfMaxDistance;
    }


public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
    MapLine();

    MapLine(const Eigen::Vector3f &LsPos, const Eigen::Vector3f &LePos, const Eigen::Vector3f &LsColor, const Eigen::Vector3f &LeColor, 
         KeyFrame* pRefKF, Map* pMap);
    //MapLine(const double invDepth, cv::Point2f uv_init, KeyFrame* pRefKF, KeyFrame* pHostKF, Map* pMap);
    MapLine(const Eigen::Vector3f &LsPos, const Eigen::Vector3f &LePos,  Map* pMap, Frame* pFrame, const int &idxF);

    void SetLineWorldPos(const Eigen::Vector3f &LsPos, const Eigen::Vector3f &LePos);
    std::pair<Eigen::Vector3f, Eigen::Vector3f> GetLineWorldPos();  //get Line start and end point

    void SetLineColorRGB(const Eigen::Vector3f &LsColor, const Eigen::Vector3f &LeColor);
    std::pair<Eigen::Vector3f, Eigen::Vector3f> GetLineColorRGB();

    void setRetrived(const bool retrived);
    bool isRetrived();

    std::pair<Eigen::Vector3f, Eigen::Vector3f> GetLineNormal();
    //void SetLineNormalVector(const Eigen::Vector3f& Lsnormal, const Eigen::Vector3f& Lenormal);

    Eigen::Vector3f GetLineNormalVector();
    void SetLineNormalVector(const Eigen::Vector3f& normal);

    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,std::tuple<int,int>> GetLineObservations();
    int Observations();

    void AddLineObservation(KeyFrame* pKF,int idx);
    void EraseLineObservation(KeyFrame* pKF);

    std::tuple<int,int> GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapLine* pMP);    
    MapLine* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnLineFound;
    }

    void ComputeDistinctiveDescriptors();
    //void ComputePluckerLineFromWorldLine(); 
    // 在 MapLine.h 中修改为：
    void ComputePluckerLineFromWorldLine(const Eigen::Vector3f& pCamCenter = Eigen::Vector3f::Zero());  //initial, if world line end pnts exist

    void UpdateWorldEndpointsMonoFallback();
    void UpdateEndpointsFromPluckerAndObservations();

    cv::Mat GetDescriptor();
    cv::Mat GetDescriptorAt(int i);
    cv::Mat GetLineDescriptor();

    float GetObservationDepth0(KeyFrame* pKf, int idx);
    float GetObservationDepth1(KeyFrame* pKf, int idx);
    const std::vector<Eigen::Vector3f>& GetLineSampledPoints3D();
    const std::vector<Eigen::Vector2f>& GetLineSampledPoints2D();  //image pnts coordinate
    const std::vector<cv::Vec3b>& GetLineSampledPntsColors();

    void SetObservationLineLsDepth(KeyFrame* pKf, int idx, float dv);
    void SetObservationLineLeDepth(KeyFrame* pKf, int idx, float dv);

    void UpdateNormalAndDepth();
    void UpdateFromPluckerLineNew();

    bool HasCachedWorldObservationLineEndPoints();
    //End Point
    //bool UnprojectStereoLine(const KeyFrame* pKF,const cv::line_descriptor::KeyLine &kl,Eigen::Matrix<float,6,1> &Lw);
    //bool UnprojectStereoLinePlucker(const KeyFrame* pKF,const cv::line_descriptor::KeyLine &kl,Eigen::Matrix<float,6,1> &Lw);


    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

    Map* GetMap();
    void UpdateMap(Map* pMap);

    void PrintObservations();
    //float MapLine::GetFoundRatio()

    void PreSave(set<KeyFrame*>& spKF,set<MapLine*>& spMP);
    void PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapLine*>& mpMPid);

    Eigen::Vector3f GetProjectedLineABC(KeyFrame *pKF);

    void SetPluckerLineNew(const Eigen::Matrix<double,6,1>& plk);

    // --- Set ---
    inline void SetPluckerLine(const Eigen::Matrix<double,6,1>& Plucker)
    {
        unique_lock<mutex> lock(mMutexPos);
        // 检查方向向量 v 的模是否接近 0 (即直线退化为点)
        if (Plucker.tail<3>().norm() < 1e-6) return;
        mWorldPlucker = Plucker;
    }

    // --- Get ---
    inline Eigen::Matrix<double,6,1> GetPluckerLine()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPlucker;
    }

    //void SetPluckerLineNew(const Eigen::Matrix<double,6,1>& plk);

    // 将像素点 (u,v) 在该 keyframe 下反投影成相机坐标系的单位方向向量（未缩放，单位向量）
    // 返回：rayDir_world (单位向量), camCenter_world (相机中心在世界系)
    static void BackprojectPixelToWorldRay(KeyFrame* pKF, const cv::Point2f &uv, Eigen::Vector3f &camCenter_world, Eigen::Vector3f &rayDir_world);

    // 将 Plücker 参数 L = [n(3); v(3)] 转换为 直线上一点 p0 (world) 和方向 dir (world, 单位向量)
    // p0 = (n × v) / |v|^2
    static void PluckerToPointAndDir(const Eigen::Matrix<double,6,1> &L, Eigen::Vector3f &p0, Eigen::Vector3f &dir);

    // 计算两条无穷直线（L1: P1 + s * d1, L2: P2 + t * d2）间的最近点
    // 返回：点在 L1 上的参数 s，点在 L2 上的参数 t，以及对应的点 p1 = P1 + s*d1, p2 = P2 + t*d2
    // 若平行，会选择投影到其中一条上
    static void ClosestPointsBetweenLines(const Eigen::Vector3f &P1, const Eigen::Vector3f &d1,
                                      const Eigen::Vector3f &P2, const Eigen::Vector3f &d2,
                                      float &s_out, float &t_out,
                                      Eigen::Vector3f &p1_out, Eigen::Vector3f &p2_out);

    // 稳健均值（按每个坐标维度取中位数）
    // 输入：points (非空)
    // 返回：三维中位数向量
    static Eigen::Vector3f RobustMedian3(const std::vector<Eigen::Vector3f> &points);

    // 返回 true 表示成功拿到端点 (s,e)
    static bool GetKeyFrameLineEndpoints(KeyFrame* pKF, int lineIdx, cv::Point2f &s_out, cv::Point2f &e_out);
    
    //Important
    void UpdateFromPluckerLine();

    void UpdateWorldEndpointsFromObservationLineDepth();

    //Lw,和pnts_3d 更新mLineWorldPos
    //void UpdateWorldEndpointsFromObservationPntsAndPluckerLine(const Eigen::Matrix<double,6,1>& Lw, std::vector<Eigen::Vector3d>& pnts_3d);

    void UpdateWorldEndpointsFromObservationPntsAndPluckerLine(
        const Eigen::Matrix<double,6,1>& Lw,
        const std::vector<Eigen::Vector3d>& pnts_3d,
        double lower_q = 0.05,
        double upper_q =  0.95);

    bool  UpdatePluckerFromBackProjectLines();  //to do next

    void SamplePointsAlongLinesWorld3D(float sample_step = 0.2f);   //sample points along 3D lines（先在图像中采样2D线段，然后获取点2D图像，再投影到3D中）
    void SamplePointsAlongLinesWorld3D_old();
    void SamplePointsByImageLength(KeyFrame* pKF, float pixel_step = 0.05f);
    void SamplePointsAlongLine_MultiViewWeighted(
        float sample_step,
        float view_angle_power);
    void SamplePointsAlongLine_MultiViewWeighted_Advanced(
        float sample_step,        // 世界坐标采样步长 (e.g. 0.05f)
        float view_angle_power,   // 视角权重指数 (e.g. 2.0)
        float sigma_line_pixel,   // 图像线一致性 σ (e.g. 3.0 px)
        int   top_k               // Top-K 视角 (e.g. 3)
    );

    bool IsValidLineMultiView(float pixel_thresh);
    float ComputeMaxParallaxAngle();

    // 计算颜色均值，忽略“黑色 / 近黑色”点
    cv::Vec3b AverageColorIgnoreBlack(const std::vector<cv::Vec3b>& colors, int black_thresh = 5);

    bool ComputeLineABCFromKeyLine(const cv::line_descriptor::KeyLine& kl, float& a, float& b, float& c);

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by the tracking
    float mLsTrackProjX;
    float mLsTrackProjY;
    float mLeTrackProjX;
    float mLeTrackProjY;
    float mLsTrackDepth;
    float mLeTrackDepth;
    float mLineTrackDepth;

    float mTrackDepthR; //to do next...
    float mTrackProjXR; //to do next...
    float mTrackProjYR; //to do next...

    // TrackLocalMap - UpdateLocalLines中防止将MapLines重复添加至mvpLocalMapLines的标记
    bool mbLineTrackInView, mbLineTrackInViewR;
    int mnLineTrackScaleLevel, mnLineTrackScaleLevelR;
    float mLineTrackViewCos, mLineTrackViewCosR;
    long unsigned int mnLastFrameSeen;

    //TrackLocalMap - SearchByProjection 中决定是否对特征线进行投影匹配的参考
    //mbTrackInView == false 的点有几种：
    //1. 该线段已经和当前帧经过匹配(TrackReferenceKeyFrame, TrackWithMotionModel),但是在优化过程中被认为是外点
    //2. 该线段在当前帧视野内且进行匹配后为内点，这类点不需要再进行投影
    //3. 该线段在当前帧视野外（为通过isInFrustum的判断）
    long unsigned int mnTrackReferenceForFrame;
    
    // TrackLocalMap - SearchLocalLines中决定是否进行isInFrustum判断的变量
    // mnLastFrameSeen==mCurrentFrame.mnId的line有集中：
    // a.已经和当前帧经过匹配（TrackReferenceKeyFrame, TrackWithMotionModel)，但在优化过程中认为是外点
    // b.已经和当前帧经过匹配且为内点，这类line也不需要再进行投影

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnLoopLineForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    Eigen::Vector3f mPosGBA;
    Eigen::Vector3f mPos1GBA, mPos2GBA; // 用于全局BA的线段端点
    long unsigned int mnBAGlobalForKF;
    long unsigned int mnBALocalForMerge;

    // Variable used by merging
    Eigen::Vector3f mPosMerge;
    Eigen::Vector3f mNormalVectorMerge;

    // 新增：记录上次发送给 GS 时的长度
    float mLastSentLength = 0.0f;   // 上次发送给 GS 时的长度

    // For inverse depth optimization
    double mLsInvDepth, mLeInvDepth;
    double mLsInitU;
    double mLsInitV;
    double mLeInitU;
    double mLeInitV;
    KeyFrame* mpHostKF;

    static std::mutex mGlobalMutex;

    unsigned int mnOriginMapId;

protected:

    // 判定是否为“黑色 / 近黑色”
    inline bool IsBlackZDG(const cv::Vec3b& c, int thresh = 5)
    {
        return (c[0] <= thresh && c[1] <= thresh && c[2] <= thresh);
    }

     // Position in absolute coordinates
     Eigen::Matrix<float,6,1> mLineWorldPos;
     Eigen::Vector3f mLsWorldPos, mLeWorldPos;  //line start and end point

     std::vector<Eigen::Vector3f> mSampledPoints3D;  // sampled 3D points
     std::vector<Eigen::Vector2f> mSampledPoints2D;  // projected 2D points
     std::vector<cv::Vec3b> mSampledPointsColor;    // color of sampled 3D points

     //
     Eigen::Matrix<double,6,1> mWorldPlucker;   //plucker

     // RGB Color from the first observation
     // -- Useless for ORB-SLAM3 but useful in Gaussian Mapping, so supposed to be constant since created
     Eigen::Vector3f mLsColorRGB, mLeColorRGB;  //line start and end point color

     // Retrival flag, ever retirved by Gaussian Mapping
     bool mbRetrived;

     // Keyframes observing the Line and associated index in keyframe
     std::map<KeyFrame*,std::tuple<int,int> > mLineObservations;
     // For save relation without pointer, this is necessary for save/load function
     std::map<long unsigned int, int> mLineBackupObservationsId1;
     std::map<long unsigned int, int> mLineBackupObservationsId2;

     // Mean viewing direction(done)
     Eigen::Vector3f mLineNormalVector;

     // Best descriptor to fast matching
     cv::Mat mLineDescriptor;

     //先特征的描述子集
     std::vector<cv::Mat> mLineDescriptors;
     //每个观察线段的单位方向向量，中点
     std::vector<Eigen::Vector3f> mLineDirVectors;

     // Reference KeyFrame
     KeyFrame* mpRefKF;
     long unsigned int mBackupRefKFId;

     // Tracking counters
     int mnLineVisible;
     int mnLineFound;

     // Bad flag (we do not currently erase MapLine from memory)
     bool mbLineBad;
     MapLine* mpLineReplaced;
     // For save relation without pointer, this is necessary for save/load function
     long long int mLineBackupReplacedId;

     // Scale invariance distances
     float mfMinDistance;
     float mfMaxDistance;

     Map* mpMap;

     // Mutex
     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
     std::mutex mMutexMap;
     std::mutex mMutexRetrival;

};

} //namespace ORB_SLAM

#endif // MAPLINE_H
