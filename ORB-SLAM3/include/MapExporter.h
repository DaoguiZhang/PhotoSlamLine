#ifndef MAPEXPORTER_H
#define MAPEXPORTER_H

#include <vector>
#include <string>
#include <Eigen/Core>
#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "MapLine.h"

namespace ORB_SLAM3
{

class MapExporter
{
public:
    MapExporter() = default;

    enum class LineColorMode {
        LENGTH,
        TRACK_NUM,
        ID_HASH
    };

    inline Eigen::Vector3f ColorMapJet(float t)
    {
        // Jet colormap: t ∈ [0,1]
        t = std::max(0.f, std::min(1.f, t));
        float r = std::min(1.f, std::max(0.f, 1.5f - std::fabs(4*t - 3)));
        float g = std::min(1.f, std::max(0.f, 1.5f - std::fabs(4*t - 2)));
        float b = std::min(1.f, std::max(0.f, 1.5f - std::fabs(4*t - 1)));
        return Eigen::Vector3f(r, g, b);
    }

    /**
     * @brief 导出局部地图 + 相机 + 坐标系到 PLY 文件
     * 
     * @param F 当前帧 Frame
     * @param LocalMapPoints 局部地图点
     * @param LocalMapLines 局部地图线段
     * @param filename 输出文件名
     * @param axisLength 相机坐标系长度
     */
    static void ExportLocalMapWithCameraAxes(
        Frame &F,
        const std::vector<MapPoint*> &LocalMapPoints,
        const std::vector<MapLine*> &LocalMapLines,
        const std::string &filename = "local_map_with_camera_axes.ply",
        float axisLength = 0.2f
    );

    // 导出 OBJ 文件
    static void ExportLocalMapWithCameraAxesOBJ(
        Frame &F,
        const std::vector<MapPoint*> &LocalMapPoints,
        const std::vector<MapLine*> &LocalMapLines,
        const std::string &filename = "local_map_with_camera_axes.obj",
        float axisLength = 0.2f)
    {
        std::ofstream ofs(filename);
        if (!ofs.is_open())
        {
            std::cerr << "[MapExporter] Cannot open file: " << filename << std::endl;
            return;
        }

        size_t vertexOffset = 1; // OBJ 顶点索引从1开始

        // --- 写 MapPoints ---
        for (auto pMP : LocalMapPoints)
        {
            if (!pMP || pMP->isBad()) continue;
            Eigen::Vector3f Pw = pMP->GetWorldPos();
            ofs << "v " << Pw(0) << " " << Pw(1) << " " << Pw(2) << "\n";
        }
        size_t mpCount = vertexOffset;
        vertexOffset += LocalMapPoints.size();

        // --- 写 MapLines ---
        for (auto l : LocalMapLines)
        {
            if (!l || l->isBad()) continue;
            Eigen::Vector3f P1 = l->GetLineWorldPos().first;
            Eigen::Vector3f P2 = l->GetLineWorldPos().second;

            ofs << "v " << P1(0) << " " << P1(1) << " " << P1(2) << "\n";
            ofs << "v " << P2(0) << " " << P2(1) << " " << P2(2) << "\n";

            ofs << "l " << vertexOffset << " " << (vertexOffset + 1) << "\n";
            vertexOffset += 2;
        }

        // --- 相机中心 ---
        Eigen::Vector3f camPos = F.GetCameraCenter();
        ofs << "v " << camPos(0) << " " << camPos(1) << " " << camPos(2) << "\n";
        size_t camIdx = vertexOffset;
        vertexOffset += 1;

        // --- 相机坐标系三个轴 ---
        Eigen::Vector3f X = camPos + axisLength * F.GetRcw().row(0).transpose();
        Eigen::Vector3f Y = camPos + axisLength * F.GetRcw().row(1).transpose();
        Eigen::Vector3f Z = camPos + axisLength * F.GetRcw().row(2).transpose();

        ofs << "v " << X(0) << " " << X(1) << " " << X(2) << "\n";
        ofs << "v " << Y(0) << " " << Y(1) << " " << Y(2) << "\n";
        ofs << "v " << Z(0) << " " << Z(1) << " " << Z(2) << "\n";

        // 坐标系线段
        ofs << "l " << camIdx << " " << (camIdx + 1) << "\n"; // X轴
        ofs << "l " << camIdx << " " << (camIdx + 2) << "\n"; // Y轴
        ofs << "l " << camIdx << " " << (camIdx + 3) << "\n"; // Z轴

        ofs.close();
        std::cout << "[MapExporter] Exported OBJ: " << filename << std::endl;
    }

    // 其他导出函数可以在这里添加
    static void ExportMapPointsToOBJ(
        const std::vector<MapPoint*> &mapPoints,
        const std::string &filename = "map_points.OBJ"
    );

    static void ExportMapLinesToOBJ(
        const std::vector<MapLine*> &mapLines,
        const std::string &filename = "map_lines.OBJ");
    
    static void ExportMapPointsWithCameraAxesOBJ(
        Frame &F,
        const std::vector<MapPoint*> &mapPoints,
        const std::string &filename = "map_points_with_camera_axes.obj",
        float axisLength = 0.2f);
    
    static void ExportMapPointsWithCameraAxesOBJKeyFrame(
        KeyFrame *pKF,
        const std::vector<MapPoint*> &mapPoints,
        const std::string &filename = "map_points_with_camera_axes.obj",
        float axisLength = 0.2f);
    
    static void ExportMapLinesWithCameraAxesOBJ(
        Frame &F,
        const std::vector<MapLine*> &mapLines,
        const std::string &filename = "map_lines_with_camera_axes.obj",
        float axisLength = 0.2f);
    
    static void ExportMapLinesWithCameraAxesOBJKeyFrame(
        KeyFrame *pKF,
        const std::vector<MapLine*> &mapLines,
        const std::string &filename = "map_lines_with_camera_axes.obj",
        float axisLength = 0.2f);

    void ExportFullSceneOBJ(
        const std::vector<MapPoint*> &mapPoints,
        const std::vector<MapLine*> &mapLines,
        const std::vector<KeyFrame*> &keyframes,
        const std::string &filename = "full_scene.obj",
        float axisLength = 0.2f,
        LineColorMode lineColorMode = LineColorMode::LENGTH);
    
};

} // namespace ORB_SLAM3

#endif // MAPEXPORTER_H
