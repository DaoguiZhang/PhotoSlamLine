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

#ifndef ATLAS_H
#define ATLAS_H

#include "Map.h"
#include "MapPoint.h"
#include "MapLine.h"  //added for MapLine
#include "KeyFrame.h"
#include "GeometricCamera.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"

#include <set>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <tuple>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>


namespace ORB_SLAM3
{
class Viewer;
class Map;
class MapPoint;
class MapLine;  //added for MapLine
class KeyFrame;
class KeyFrameDatabase;
class Frame;
class KannalaBrandt8;
class Pinhole;

//BOOST_CLASS_EXPORT_GUID(Pinhole, "Pinhole")
//BOOST_CLASS_EXPORT_GUID(KannalaBrandt8, "KannalaBrandt8")

class MappingOperation
{
public:
    enum OprType{
        LocalMappingBA = 1,
        LoopClosingBA = 2,
        ScaleRefinement = 3
    };

private:
    MappingOperation(
        const MappingOperation &opr,
        const std::lock_guard<std::mutex> &,
        const std::lock_guard<std::mutex> &,
        const std::lock_guard<std::mutex> &)
        : mvAssociatedKeyFrames(std::move(opr.mvAssociatedKeyFrames)),
          mvAssociatedMapPoints(std::move(opr.mvAssociatedMapPoints)),
          mvAssociatedMapLines(std::move(opr.mvAssociatedMapLines)),
          meOperationType(opr.meOperationType),
          mfScale(opr.mfScale),
          mT(opr.mT)
    {}

public:
    MappingOperation(
        OprType type,
        const float scale = 1.0f,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()),
        const std::size_t nKFs = 0UL,
        const std::size_t nMPs = 0UL)
        : meOperationType(type),
          mfScale(scale),
          mT(T)
    {
        mvAssociatedKeyFrames.reserve(nKFs);
        int length = nMPs * 3;
        std::get<0>(mvAssociatedMapPoints).reserve(length);
        std::get<1>(mvAssociatedMapPoints).reserve(length);

        // MapLines 初始化
        std::get<0>(mvAssociatedMapLines).reserve(length); // 端点
        std::get<1>(mvAssociatedMapLines).reserve(length); // 颜色
        // Sampled Points 初始化 (假设每个线采样10个点)
        std::get<0>(mvAssociatedLineSampledPoints).reserve(length * 10); 
        std::get<1>(mvAssociatedLineSampledPoints).reserve(length * 10);
        std::get<2>(mvAssociatedLineSampledPoints).reserve(length * 10 / 3); // 存 line_dir
        // [新增] MapLines ID 初始化
        mvAssociatedMapLineIds.reserve(length); // 估算一下，或者由 reserveMapLines 处理
    }

    MappingOperation(const MappingOperation &opr)
        : MappingOperation(
            opr,
            std::lock_guard<std::mutex>(opr.mMutexKeyFrames),
            std::lock_guard<std::mutex>(opr.mMutexMapPoints),
            std::lock_guard<std::mutex>(opr.mMutexMapLines))
    {}

public:
    void reserveKeyFrames(const std::size_t nKFs)
    {
        mvAssociatedKeyFrames.reserve(nKFs);
    }

    void addKeyFrame(KeyFrame* pKF, bool isLoopClosureKF = false)
    {
        std::unique_lock<std::mutex> lock(mMutexKeyFrames);
        std::vector<float> pixels;
        std::vector<float> pointsLocal;
        std::vector<float> keylinePixels; // added for keyline
        std::vector<float> keylinePointsLocal; // added for keyline(important, 明天要改这部分的内容，它关于后续的内容，需要图像的线段和世界坐标系的线段对应，一起传输出去，放到mvAssociatedKeyFrames中)
        pKF->GetKeypointInfo(pixels, pointsLocal);
        pKF->GetKeyLineInfo(keylinePixels, keylinePointsLocal); // added for keyline
        mvAssociatedKeyFrames.emplace_back(
            std::make_tuple(
                pKF->mnId,
                pKF->mpCamera->GetId(),
                pKF->GetPose(),
                pKF->imgLeftRGB.clone(),
                isLoopClosureKF,
                pKF->imgAuxiliary,
                pixels,
                pointsLocal,
                pKF->mNameFile,
                keylinePixels)); // added for keyline
    }

    std::vector<std::tuple<
        unsigned long,
        unsigned long,
        Sophus::SE3f,
        cv::Mat,
        bool,
        cv::Mat,
        std::vector<float>,
        std::vector<float>,
        std::string,
        std::vector<float>/*keyline pixel*/>>&
    associatedKeyFrames() { return mvAssociatedKeyFrames; }

    void reserveMapPoints(const std::size_t nMPs)
    {
        int length = nMPs * 3;
        std::get<0>(mvAssociatedMapPoints).reserve(length);
        std::get<1>(mvAssociatedMapPoints).reserve(length);
    }

    void reserveMapLines(const std::size_t nMLs) // added for MapLine
    {
        int length = nMLs * 6;
        std::get<0>(mvAssociatedMapLines).reserve(length);
        std::get<1>(mvAssociatedMapLines).reserve(length);
        // [新增]
        mvAssociatedMapLineIds.reserve(nMLs);
    }

    void addMapPoint(MapPoint* pMP)
    {
        std::unique_lock<std::mutex> lock(mMutexMapPoints);
        auto pt = pMP->GetWorldPos();
        std::get<0>(mvAssociatedMapPoints).emplace_back(pt.x());
        std::get<0>(mvAssociatedMapPoints).emplace_back(pt.y());
        std::get<0>(mvAssociatedMapPoints).emplace_back(pt.z());
        auto color = pMP->GetColorRGB();
        std::get<1>(mvAssociatedMapPoints).emplace_back(color.x());
        std::get<1>(mvAssociatedMapPoints).emplace_back(color.y());
        std::get<1>(mvAssociatedMapPoints).emplace_back(color.z());
    }

    void addMapLine(MapLine* pML) 
    {
        if(!pML || pML->isBad()) return;

        std::unique_lock<std::mutex> lock(mMutexMapLines);

        // [新增 1] 去重检查：如果这次打包已经包含这个线段，直接跳过
        if (msInsertedLineIds.count(pML->mnId)) {
            return; 
        }
        msInsertedLineIds.insert(pML->mnId);

        // [新增 2] 存储 ID
        mvAssociatedMapLineIds.push_back(pML->mnId);

        // --- 以下是原有的几何存储逻辑 ---
        auto pt1 = pML->GetLineWorldPos().first;
        auto pt2 = pML->GetLineWorldPos().second;
        
        std::get<0>(mvAssociatedMapLines).emplace_back(pt1.x());
        std::get<0>(mvAssociatedMapLines).emplace_back(pt1.y());
        std::get<0>(mvAssociatedMapLines).emplace_back(pt1.z());
        std::get<0>(mvAssociatedMapLines).emplace_back(pt2.x());
        std::get<0>(mvAssociatedMapLines).emplace_back(pt2.y());
        std::get<0>(mvAssociatedMapLines).emplace_back(pt2.z());
        
        auto color1 = pML->GetLineColorRGB().first;
        std::get<1>(mvAssociatedMapLines).emplace_back(color1.x());
        std::get<1>(mvAssociatedMapLines).emplace_back(color1.y());
        std::get<1>(mvAssociatedMapLines).emplace_back(color1.z());
        auto color2 = pML->GetLineColorRGB().second;
        std::get<1>(mvAssociatedMapLines).emplace_back(color2.x());
        std::get<1>(mvAssociatedMapLines).emplace_back(color2.y());
        std::get<1>(mvAssociatedMapLines).emplace_back(color2.z());

        // --- Sampled Points 处理 ---
        // 这里的逻辑你是对的：直接取 MapLine 里计算好的点
        const auto& sampled_pts = pML->GetLineSampledPoints3D(); 
        const auto& sampled_cols = pML->GetLineSampledPntsColors();

        // 简单的安全检查
        if(sampled_pts.empty()) return; 

        // 计算方向 (用于 Gaussian 旋转)
        Eigen::Vector3f dir = (pt2 - pt1).normalized();
        
        // 如果 dir 出现 NaN (例如点重合)，做个保护
        if (!std::isfinite(dir.x())) dir = Eigen::Vector3f::UnitX();

        for(size_t i=0; i<sampled_pts.size(); ++i) {
            // Pos
            std::get<0>(mvAssociatedLineSampledPoints).emplace_back(sampled_pts[i].x());
            std::get<0>(mvAssociatedLineSampledPoints).emplace_back(sampled_pts[i].y());
            std::get<0>(mvAssociatedLineSampledPoints).emplace_back(sampled_pts[i].z());
            
            // Color (保护数组越界，以防万一)
            if (i < sampled_cols.size()) {
                std::get<1>(mvAssociatedLineSampledPoints).emplace_back(sampled_cols[i][0] / 255.0f);
                std::get<1>(mvAssociatedLineSampledPoints).emplace_back(sampled_cols[i][1] / 255.0f);
                std::get<1>(mvAssociatedLineSampledPoints).emplace_back(sampled_cols[i][2] / 255.0f);
            } else {
                // Fallback color
                std::get<1>(mvAssociatedLineSampledPoints).emplace_back(1.0f);
                std::get<1>(mvAssociatedLineSampledPoints).emplace_back(1.0f);
                std::get<1>(mvAssociatedLineSampledPoints).emplace_back(1.0f);
            }

            // Direction
            std::get<2>(mvAssociatedLineSampledPoints).emplace_back(dir.x());
            std::get<2>(mvAssociatedLineSampledPoints).emplace_back(dir.y());
            std::get<2>(mvAssociatedLineSampledPoints).emplace_back(dir.z());
        }
    }

    #if 0
    void addMapLine(MapLine* pML) // added for MapLine
    {
        std::unique_lock<std::mutex> lock(mMutexMapLines);
        auto pt1 = pML->GetLineWorldPos().first;
        auto pt2 = pML->GetLineWorldPos().second;
        std::get<0>(mvAssociatedMapLines).emplace_back(pt1.x());
        std::get<0>(mvAssociatedMapLines).emplace_back(pt1.y());
        std::get<0>(mvAssociatedMapLines).emplace_back(pt1.z());
        std::get<0>(mvAssociatedMapLines).emplace_back(pt2.x());
        std::get<0>(mvAssociatedMapLines).emplace_back(pt2.y());
        std::get<0>(mvAssociatedMapLines).emplace_back(pt2.z());
        auto color1 = pML->GetLineColorRGB().first;
        std::get<1>(mvAssociatedMapLines).emplace_back(color1.x());
        std::get<1>(mvAssociatedMapLines).emplace_back(color1.y());
        std::get<1>(mvAssociatedMapLines).emplace_back(color1.z());
        auto color2 = pML->GetLineColorRGB().second;
        std::get<1>(mvAssociatedMapLines).emplace_back(color2.x());
        std::get<1>(mvAssociatedMapLines).emplace_back(color2.y());
        std::get<1>(mvAssociatedMapLines).emplace_back(color2.z());

        // 2. 存储采样点 (Sampled Points) - 用于 Gaussian 初始化
        // 必须假设 pML 已经完成了采样计算
        const auto& sampled_pts = pML->GetLineSampledPoints3D(); 
        const auto& sampled_cols = pML->GetLineSampledPntsColors();

        if(sampled_pts.empty() || sampled_cols.empty()) {
            std::cerr << "Warning: MapLine has no sampled points or colors!" << std::endl;
            return;
        }

        if(sampled_pts.size() != sampled_cols.size()) {
            std::cerr << "Error: Sampled points and colors size mismatch!" << std::endl;
            return;
        }

        // 计算线方向 (用于 Gaussian 旋转初始化)
        Eigen::Vector3f dir = (pt2 - pt1).normalized();
        for(size_t i=0; i<sampled_pts.size(); ++i) {
            // Pos
            std::get<0>(mvAssociatedLineSampledPoints).emplace_back(sampled_pts[i].x());
            std::get<0>(mvAssociatedLineSampledPoints).emplace_back(sampled_pts[i].y());
            std::get<0>(mvAssociatedLineSampledPoints).emplace_back(sampled_pts[i].z());
            // Color (归一化到 0-1 float)
            std::get<1>(mvAssociatedLineSampledPoints).emplace_back(sampled_cols[i][0] / 255.0f);
            std::get<1>(mvAssociatedLineSampledPoints).emplace_back(sampled_cols[i][1] / 255.0f);
            std::get<1>(mvAssociatedLineSampledPoints).emplace_back(sampled_cols[i][2] / 255.0f);
            // Direction (新增：为了初始化 Line Gaussian 的旋转)
            std::get<2>(mvAssociatedLineSampledPoints).emplace_back(dir.x());
            std::get<2>(mvAssociatedLineSampledPoints).emplace_back(dir.y());
            std::get<2>(mvAssociatedLineSampledPoints).emplace_back(dir.z());
        }
    }
    #endif

    // Getter for Sampled Points (Pos, Color, Direction)
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>& associatedLineSampledPoints() { 
        return mvAssociatedLineSampledPoints; 
    }

    // Getter for Map Points (Pos, Color)
    std::tuple<std::vector<float/*pos*/>, std::vector<float/*color*/>>&
    associatedMapPoints() { return mvAssociatedMapPoints; }

    // Getter for Map Lines (Pos, Color)
    std::tuple<std::vector<float/*pos*/>, std::vector<float/*color*/>>&
    associatedMapLines() { return mvAssociatedMapLines; } // added for MapLine

    // [新增] 获取 MapLine IDs
    std::vector<unsigned long>& associatedMapLineIds() {
        return mvAssociatedMapLineIds;
    }

public:
    // Type
    OprType meOperationType;

    // Data
    float mfScale; ///<  ScaleRefinement: global; LoopClosingBA: only for visible; LocalMappingBA: meaningless
    float line_sample_distance_ = 0.05f; // added by zdg
    Sophus::SE3f mT;

protected:
    // Data
    std::tuple<std::vector<float/*pos*/>,
               std::vector<float/*color*/>> mvAssociatedMapPoints;

    // [新增] 用于存储 MapLine ID，以便接收端知道是更新哪个线段
    std::vector<unsigned long> mvAssociatedMapLineIds; 

    // [新增] 内部辅助变量，防止同一个 MappingOperation 里重复添加同一个线段
    std::unordered_set<unsigned long> msInsertedLineIds;
    
    // [Pos (x1,y1,z1, x2,y2,z2...), Color (r1,g1,b1, r2,g2,b2...)]
    std::tuple<std::vector<float/*pos*/>,
               std::vector<float/*color*/>> mvAssociatedMapLines; // added for MapLine
    
    // [Pos, Color, Direction] - 纯数据存储，解耦 MapLine 对象
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> mvAssociatedLineSampledPoints;

    std::vector<std::tuple<
        unsigned long/*Id*/,
        unsigned long/*CameraId*/,
        Sophus::SE3f/*pose*/,
        cv::Mat/*image*/,
        bool/*isLoopClosure*/,
        cv::Mat/*auxiliaryImage*/,
        std::vector<float>/*keypoints pixel*/,
        std::vector<float>/*keypoints local 3D*/,
        std::string/*main image file name*/,
        std::vector<float>/*keyline pixel*/>> mvAssociatedKeyFrames;

    // Mutex
    mutable std::mutex mMutexMapPoints;
    mutable std::mutex mMutexMapLines; // added for MapLine
    mutable std::mutex mMutexKeyFrames;
};

class Atlas
{
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar.template register_type<Pinhole>();
        ar.template register_type<KannalaBrandt8>();

        // Save/load a set structure, the set structure is broken in libboost 1.58 for ubuntu 16.04, a vector is serializated
        //ar & mspMaps;
        ar & mvpBackupMaps;
        ar & mvpCameras;
        // Need to save/load the static Id from Frame, KeyFrame, MapPoint and Map
        ar & Map::nNextId;
        ar & Frame::nNextId;
        ar & KeyFrame::nNextId;
        ar & MapPoint::nNextId;
        ar & GeometricCamera::nNextId;
        ar & mnLastInitKFidMap;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    Atlas();
    Atlas(int initKFid); // When its initialization the first map is created
    ~Atlas();

    void CreateNewMap();
    void ChangeMap(Map* pMap);

    unsigned long int GetLastInitKFid();

    void SetViewer(Viewer* pViewer);

    // Method for change components in the current map
    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void AddMapLine(MapLine* pML);
    //void EraseMapPoint(MapPoint* pMP);
    //void EraseKeyFrame(KeyFrame* pKF);

    GeometricCamera* AddCamera(GeometricCamera* pCam);
    std::vector<GeometricCamera*> GetAllCameras();

    /* All methods without Map pointer work on current map */
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    /*all maplines on current map*/
    void SetReferenceMapLines(const std::vector<MapLine*> &vpMLs);

    long unsigned int MapPointsInMap();
    long unsigned KeyFramesInMap();
    long unsigned int MapLinesInMap();

    // Method for get data in current map
    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();
    std::unordered_set<unsigned long> GetCurrentKeyFrameIds();

    std::vector<MapLine*> GetAllMapLines();  //added for MapLine
    std::vector<MapLine*> GetReferenceMapLines();  //added for MapLine

    //Method for get all maps in the atlas
    vector<Map*> GetAllMaps();

    int CountMaps();

    void clearMap();

    void clearAtlas();

    Map* GetCurrentMap();

    void SetMapBad(Map* pMap);
    void RemoveBadMaps();

    bool isInertial();
    void SetInertialSensor();
    void SetImuInitialized();
    bool isImuInitialized();

    // Function for garantee the correction of serialization of this object
    void PreSave();
    void PostLoad();

    map<long unsigned int, KeyFrame*> GetAtlasKeyframes();

    void SetKeyFrameDababase(KeyFrameDatabase* pKFDB);
    KeyFrameDatabase* GetKeyFrameDatabase();

    void SetORBVocabulary(ORBVocabulary* pORBVoc);
    ORBVocabulary* GetORBVocabulary();

    long unsigned int GetNumLivedKF();

    long unsigned int GetNumLivedMP();

    // Mapping operations queue, important for Gaussian splatting module(inteface between slam results and Gaussian splatting module)
    void pushMappingOperation(MappingOperation opr);
    MappingOperation getAndPopMappingOperation();
    bool hasMappingOperation();
    void clearMappingOperation();

protected:
    std::queue<MappingOperation> mqMappingOperations;

    std::set<Map*> mspMaps;
    std::set<Map*> mspBadMaps;
    // Its necessary change the container from set to vector because libboost 1.58 and Ubuntu 16.04 have an error with this cointainer
    std::vector<Map*> mvpBackupMaps;

    Map* mpCurrentMap;

    std::vector<GeometricCamera*> mvpCameras;

    unsigned long int mnLastInitKFidMap;

    Viewer* mpViewer;
    bool mHasViewer;

    // Class references for the map reconstruction from the save file
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    // Mutex
    std::mutex mMutexAtlas;
    std::mutex mMutexMappingOperations;

}; // class Atlas

} // namespace ORB_SLAM3

#endif // ATLAS_H
