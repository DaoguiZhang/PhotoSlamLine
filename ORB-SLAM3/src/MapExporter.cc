#include "MapExporter.h"
#include <fstream>
#include <iostream>

namespace ORB_SLAM3
{
    void MapExporter::ExportLocalMapWithCameraAxes(
        Frame &F,
        const std::vector<MapPoint*> &LocalMapPoints,
        const std::vector<MapLine*> &LocalMapLines,
        const std::string &filename,
        float axisLength)
    {
        std::ofstream ofs(filename);
        if (!ofs.is_open())
        {
            std::cerr << "[MapExporter] Cannot open file: " << filename << std::endl;
            return;
        }

        // --- 计算顶点数量 ---
        size_t numVertices = 0;
        numVertices += LocalMapPoints.size();
        for (auto l : LocalMapLines)
            if (l && !l->isBad()) numVertices += 2;
        numVertices += 1; // 相机中心
        numVertices += 3; // 坐标系三个轴

        // --- 计算边数量 ---
        size_t numEdges = 0;
        for (auto l : LocalMapLines)
            if (l && !l->isBad()) numEdges += 1;
        numEdges += 3; // 坐标系三条边

        // --- 写 PLY 头 ---
        ofs << "ply\n";
        ofs << "format ascii 1.0\n";
        ofs << "element vertex " << numVertices << "\n";
        ofs << "property float x\n";
        ofs << "property float y\n";
        ofs << "property float z\n";
        ofs << "property uchar red\n";
        ofs << "property uchar green\n";
        ofs << "property uchar blue\n";
        ofs << "element edge " << numEdges << "\n";
        ofs << "property int vertex1\n";
        ofs << "property int vertex2\n";
        ofs << "end_header\n";
        // --- 写 MapPoints (绿色) ---
        for (auto pMP : LocalMapPoints)
        {
            if (!pMP || pMP->isBad()) continue;
            Eigen::Vector3f Pw = pMP->GetWorldPos();
            ofs << Pw(0) << " " << Pw(1) << " " << Pw(2) << " 0 255 0\n";
        }

        // --- 写 MapLines (红色) ---
        size_t vertexOffset = LocalMapPoints.size();
        for (auto l : LocalMapLines)
        {
            if (!l || l->isBad()) continue;
            Eigen::Vector3f P1 = l->GetLineWorldPos().first;
            Eigen::Vector3f P2 = l->GetLineWorldPos().second;
            ofs << P1(0) << " " << P1(1) << " " << P1(2) << " 255 0 0\n";
            ofs << P2(0) << " " << P2(1) << " " << P2(2) << " 255 0 0\n";
            ofs << vertexOffset << " " << (vertexOffset + 1) << "\n";
            vertexOffset += 2;
        }
        // --- 相机中心 (黄色) ---
        Eigen::Vector3f camPos = F.GetCameraCenter();
        ofs << camPos(0) << " " << camPos(1) << " " << camPos(2) << " 255 255 0\n";
        size_t camIdx = vertexOffset;
        vertexOffset += 1;
        // --- 相机坐标系三个轴 (RGB) ---
        Eigen::Vector3f X = camPos + axisLength * F.GetRcw().row(0).transpose();
        Eigen::Vector3f Y = camPos + axisLength * F.GetRcw().row(1).transpose();
        Eigen::Vector3f Z = camPos + axisLength * F.GetRcw().row(2).transpose();
        ofs << X(0) << " " << X(1) << " " << X(2) << " 255 0 0\n";
        ofs << Y(0) << " " << Y(1) << " " << Y(2) << " 0 255 0\n";
        ofs << Z(0) << " " << Z(1) << " " << Z(2) << " 0 0 255\n";

        // 坐标系边
        ofs << camIdx << " " << (camIdx + 1) << "\n";
        ofs << camIdx << " " << (camIdx + 2) << "\n";
        ofs << camIdx << " " << (camIdx + 3) << "\n";

        ofs.close();
        std::cout << "[MapExporter] Exported local map with camera axes to "
                << filename << " | vertices: " << numVertices
                << ", edges: " << numEdges << std::endl;
    }

    void MapExporter::ExportMapPointsToOBJ(
        const std::vector<MapPoint*> &mapPoints,
        const std::string &filename) 
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[ExportMapPointsToOBJ] Error: cannot open file " << filename << std::endl;
            return;
        }
        file << "# Map points exported as OBJ format with color\n";
        file << "# Total points: " << mapPoints.size() << "\n";
        int validCount = 0;
        for (const auto &pMP : mapPoints) {
            if (!pMP) continue;
            if (pMP->isBad()) continue;  // 如果有此函数
            Eigen::Vector3f pos = pMP->GetWorldPos();
            // 检查 NaN 或无效数据
            if (!pos.allFinite()) continue;
            // 判断数据类型
            float x, y, z;
            x = pos(0);
            y = pos(1);
            z = pos(2);
            // 获取颜色信息（假设为 BGR）
            Eigen::Vector3f color(255, 255, 255);  // 默认白色
            try {
                color = pMP->GetColorRGB();
            } catch (...) {
            // 如果没有颜色信息则忽略
            }
            // 转为 [0,1] RGB
            float r = color[2];
            float g = color[1];
            float b = color[0];
            // 输出格式: v x y z r g b
            file << "v " << x << " " << y << " " << z << " "
                << r << " " << g << " " << b << "\n";
            validCount++;
        }
        file.close();
        std::cout << "[ExportMapPointsToOBJ] Exported " << validCount 
                 << " valid points to " << filename << std::endl;
    }

    void MapExporter::ExportMapLinesToOBJ(
        const std::vector<MapLine*> &mapLines,
        const std::string &filename)
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[ExportMapLinesToOBJ] Error: cannot open file " << filename << std::endl;
            return;
        }
        file << "# Map lines exported as OBJ format\n";
        file << "# Total lines: " << mapLines.size() << "\n";
        int vertexCount = 0;
        int lineCount = 0;
        // 写入顶点与线条
        for (const auto &l : mapLines) {
            if (!l) continue;
            if (l->isBad()) continue;  // 若 MapLine 定义中有此方法
            auto linePos = l->GetLineWorldPos();
            Eigen::Vector3f P1 = linePos.first;
            Eigen::Vector3f P2 = linePos.second;
            // 检查 NaN 或无效数据
            if (!P1.allFinite() || !P2.allFinite()) continue;
            // 写两个顶点
            file << "v " << P1.x() << " " << P1.y() << " " << P1.z() << "\n";
            file << "v " << P2.x() << " " << P2.y() << " " << P2.z() << "\n";
            // 写一条线：OBJ 索引从 1 开始
            file << "l " << vertexCount + 1 << " " << vertexCount + 2 << "\n";
            vertexCount += 2;
            lineCount++;
        }
        file.close();
        std::cout << "[ExportMapLinesToOBJ] Exported " << lineCount
                << " lines (" << vertexCount << " vertices) to "
                << filename << std::endl;
    }


    void MapExporter::ExportMapPointsWithCameraAxesOBJ(
        Frame &F,
        const std::vector<MapPoint*> &mapPoints,
        const std::string &filename,
        float axisLength)
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[ExportMapPointsWithCameraAxesOBJ] Error: cannot open file " 
                    << filename << std::endl;
            return;
        }
        file << "# Map points + camera axes exported as OBJ format\n";
        file << "# Total map points: " << mapPoints.size() << "\n";
        int vertexCount = 0;
        // --- 导出所有地图点 ---
        for (const auto &pMP : mapPoints) {
            if (!pMP) continue;
            if (pMP->isBad()) continue;
            Eigen::Vector3f Pw = pMP->GetWorldPos();
            if (!Pw.allFinite()) continue;
            // 若有颜色
            Eigen::Vector3f color(1.0f, 1.0f, 1.0f);  // 默认白色
            try {
                color = pMP->GetColorRGB();
            } catch (...) {}
            file << "v " << Pw.x() << " " << Pw.y() << " " << Pw.z() << " "
                << color.x() << " " << color.y() << " " << color.z() << "\n";
            vertexCount++;
        }
        // --- 相机坐标系 ---
        Eigen::Vector3f camPos = F.GetCameraCenter();
        Eigen::Matrix3f Rcw = F.GetRcw();
        Eigen::Vector3f X = camPos + axisLength * Rcw.row(0).transpose();
        Eigen::Vector3f Y = camPos + axisLength * Rcw.row(1).transpose();
        Eigen::Vector3f Z = camPos + axisLength * Rcw.row(2).transpose();
        // 相机中心（黑）
        file << "v " << camPos.x() << " " << camPos.y() << " " << camPos.z() << " 0 0 0\n";
        int camIdx = ++vertexCount;
        // X轴（红）
        file << "v " << X.x() << " " << X.y() << " " << X.z() << " 1 0 0\n";
        file << "l " << camIdx << " " << camIdx + 1 << "\n";
        vertexCount++;
        // Y轴（绿）
        file << "v " << Y.x() << " " << Y.y() << " " << Y.z() << " 0 1 0\n";
        file << "l " << camIdx << " " << camIdx + 2 << "\n";
        vertexCount++;
        // Z轴（蓝）
        file << "v " << Z.x() << " " << Z.y() << " " << Z.z() << " 0 0 1\n";
        file << "l " << camIdx << " " << camIdx + 3 << "\n";
        vertexCount++;
        file.close();
        std::cout << "[ExportMapPointsWithCameraAxesOBJ] Exported " << mapPoints.size()
                << " map points and camera axes to " << filename << std::endl;
    }

    void MapExporter::ExportMapLinesWithCameraAxesOBJ(
        Frame &F,
        const std::vector<MapLine*> &mapLines,
        const std::string &filename,
        float axisLength) 
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[ExportMapLinesWithCameraAxesOBJ] Error: cannot open file "
                    << filename << std::endl;
            return;
        }
        file << "# Map lines + camera axes exported as OBJ format\n";
        file << "# Total lines: " << mapLines.size() << "\n";
        int vertexCount = 0;
        int lineCount = 0;
        // --- 导出线段 ---
        for (const auto &l : mapLines) {
            if (!l) continue;
            if (l->isBad()) continue;
            auto linePos = l->GetLineWorldPos();
            Eigen::Vector3f P1 = linePos.first;
            Eigen::Vector3f P2 = linePos.second;
            if (!P1.allFinite() || !P2.allFinite()) continue;
            // 写入两个顶点
            file << "v " << P1.x() << " " << P1.y() << " " << P1.z() << "\n";
            file << "v " << P2.x() << " " << P2.y() << " " << P2.z() << "\n";
            // 写入一条线（OBJ 索引从 1 开始）
            file << "l " << vertexCount + 1 << " " << vertexCount + 2 << "\n";
            vertexCount += 2;
            lineCount++;
        }
        // --- 相机坐标系 ---
        Eigen::Vector3f camPos = F.GetCameraCenter();
        Eigen::Matrix3f Rcw = F.GetRcw();
        Eigen::Vector3f X = camPos + axisLength * Rcw.row(0).transpose();
        Eigen::Vector3f Y = camPos + axisLength * Rcw.row(1).transpose();
        Eigen::Vector3f Z = camPos + axisLength * Rcw.row(2).transpose();
        // 相机中心（黑）
        file << "v " << camPos.x() << " " << camPos.y() << " " << camPos.z() << " 0 0 0\n";
        int camIdx = ++vertexCount;
        // X轴（红）
        file << "v " << X.x() << " " << X.y() << " " << X.z() << " 1 0 0\n";
        file << "l " << camIdx << " " << camIdx + 1 << "\n";
        vertexCount++;
        // Y轴（绿）
        file << "v " << Y.x() << " " << Y.y() << " " << Y.z() << " 0 1 0\n";
        file << "l " << camIdx << " " << camIdx + 2 << "\n";
        vertexCount++;
        // Z轴（蓝）
        file << "v " << Z.x() << " " << Z.y() << " " << Z.z() << " 0 0 1\n";
        file << "l " << camIdx << " " << camIdx + 3 << "\n";
        vertexCount++;
        file.close();
        std::cout << "[ExportMapLinesWithCameraAxesOBJ] Exported "
                << lineCount << " map lines and camera axes to "
                << filename << std::endl;
    }

} // namespace ORB_SLAM3