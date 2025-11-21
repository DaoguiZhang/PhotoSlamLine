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

    void MapExporter::ExportMapPointsWithCameraAxesOBJKeyFrame(
        KeyFrame *pKF,
        const std::vector<MapPoint*> &mapPoints,
        const std::string &filename,
        float axisLength)
    {
        if (!pKF) {
            std::cerr << "[ExportMapPointsWithCameraAxesOBJKeyFrame] Error: KeyFrame is null\n";
            return;
        }
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[ExportMapPointsWithCameraAxesOBJKeyFrame] Error: cannot open file "
                    << filename << std::endl;
            return;
        }
        file << "# Map points + keyframe camera axes exported as OBJ format\n";
        file << "# Total map points: " << mapPoints.size() << "\n";
        int vertexCount = 0;
        // --- 导出所有地图点 ---
        for (const auto &pMP : mapPoints) {
            if (!pMP) continue;
            if (pMP->isBad()) continue;
            Eigen::Vector3f Pw = pMP->GetWorldPos();
            if (!Pw.allFinite()) continue;
            // 默认白色
            Eigen::Vector3f color(1.0f, 1.0f, 1.0f);
            try {
                color = pMP->GetColorRGB();
            } catch (...) {}
            file << "v " << Pw.x() << " " << Pw.y() << " " << Pw.z() << " "
                << color.x() << " " << color.y() << " " << color.z() << "\n";
            vertexCount++;
        }
        // --- KeyFrame 的相机坐标系 ---
        Eigen::Vector3f camPos = pKF->GetCameraCenter();
        Sophus::SE3f Tcw = pKF->GetPose();
        Eigen::Matrix4f Tcw_eign = Tcw.matrix();
        Eigen::Matrix3f Rcw = Tcw_eign.block<3,3>(0,0);
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
        std::cout << "[ExportMapPointsWithCameraAxesOBJKeyFrame] Exported "
                << mapPoints.size() << " map points and camera axes to "
                << filename << std::endl;
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


    void MapExporter::ExportMapLinesWithCameraAxesOBJKeyFrame(
        KeyFrame *pKF,
        const std::vector<MapLine*> &mapLines,
        const std::string &filename,
        float axisLength)
    {
        if (!pKF) {
            std::cerr << "[ExportMapLinesWithCameraAxesOBJKeyFrame] Error: KeyFrame is null\n";
            return;
        }
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[ExportMapLinesWithCameraAxesOBJKeyFrame] Error: cannot open file "
                    << filename << std::endl;
            return;
        }
    file << "# Map lines + keyframe camera axes exported as OBJ format\n";
    file << "# Total lines: " << mapLines.size() << "\n";

    int vertexCount = 0;
    int lineCount   = 0;

    // --- 导出 MapLines ---
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

    // --- KeyFrame 的相机坐标系 ---
    Eigen::Vector3f camPos = pKF->GetCameraCenter();
    Sophus::SE3f Tcw = pKF->GetPose();
    Eigen::Matrix4f Tcw_eign = Tcw.matrix();
    Eigen::Matrix3f Rcw = Tcw_eign.block<3,3>(0,0);

    Eigen::Vector3f X = camPos + axisLength * Rcw.row(0).transpose();
    Eigen::Vector3f Y = camPos + axisLength * Rcw.row(1).transpose();
    Eigen::Vector3f Z = camPos + axisLength * Rcw.row(2).transpose();

    // 相机中心（黑色）
    file << "v " << camPos.x() << " " << camPos.y() << " " << camPos.z() << " 0 0 0\n";
    int camIdx = ++vertexCount;

    // X 轴（红色）
    file << "v " << X.x() << " " << X.y() << " " << X.z() << " 1 0 0\n";
    file << "l " << camIdx << " " << camIdx + 1 << "\n";
    vertexCount++;

    // Y 轴（绿色）
    file << "v " << Y.x() << " " << Y.y() << " " << Y.z() << " 0 1 0\n";
    file << "l " << camIdx << " " << camIdx + 2 << "\n";
    vertexCount++;

    // Z 轴（蓝色）
    file << "v " << Z.x() << " " << Z.y() << " " << Z.z() << " 0 0 1\n";
    file << "l " << camIdx << " " << camIdx + 3 << "\n";
    vertexCount++;

    file.close();

    std::cout << "[ExportMapLinesWithCameraAxesOBJKeyFrame] Exported "
              << lineCount << " map lines and camera axes to "
              << filename << std::endl;
}

void MapExporter::ExportFullSceneOBJ(
    const std::vector<MapPoint*> &mapPoints,
    const std::vector<MapLine*> &mapLines,
    const std::vector<KeyFrame*> &keyframes,
    const std::string &filename,
    float axisLength,
    LineColorMode lineColorMode)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ExportFullSceneOBJ] Cannot open " << filename << "\n";
        return;
    }

    file << "# Full SLAM Scene Export\n";
    file << "# MapPoints: " << mapPoints.size() << "\n";
    file << "# MapLines: " << mapLines.size() << "\n";
    file << "# KeyFrames: " << keyframes.size() << "\n\n";

    int vertexCount = 0;

    // ============================================================
    // 1. MapPoints (white or stored color)
    // ============================================================
    for (auto pMP : mapPoints) {
        if (!pMP || pMP->isBad()) continue;
        Eigen::Vector3f Pw = pMP->GetWorldPos();
        if (!Pw.allFinite()) continue;

        Eigen::Vector3f col(1,1,1);
        try { col = pMP->GetColorRGB(); } catch (...) {}

        file << "v " << Pw.x() << " " << Pw.y() << " " << Pw.z() << " "
             << col.x() << " " << col.y() << " " << col.z() << "\n";
        vertexCount++;
    }

    // ============================================================
    // 2. MapLines with auto color coding
    // ============================================================
    auto getLineColor = [&](MapLine *l)->Eigen::Vector3f {

        if (lineColorMode == LineColorMode::LENGTH) {
            auto p = l->GetLineWorldPos();
            float len = (p.first - p.second).norm();
            float t = std::min(len / 2.0f, 1.0f);  // normalize, assume 2m upper
            return ColorMapJet(t);
        }
        else if (lineColorMode == LineColorMode::TRACK_NUM) {
            int track = l->Observations();  // 假设有这个 API
            float t = std::min(track / 20.f, 1.f);
            return ColorMapJet(t);
        }
        else { // ID_HASH
            unsigned int h = std::hash<MapLine*>()(l);
            float r = ((h >> 16) & 255) / 255.0f;
            float g = ((h >> 8) & 255) / 255.0f;
            float b = (h & 255) / 255.0f;
            return {r, g, b};
        }
    };

    for (auto l : mapLines) {
        if (!l || l->isBad()) continue;
        auto P = l->GetLineWorldPos();
        Eigen::Vector3f P1 = P.first;
        Eigen::Vector3f P2 = P.second;
        if (!P1.allFinite() || !P2.allFinite()) continue;

        Eigen::Vector3f col = getLineColor(l);

        file << "v " << P1.x() << " " << P1.y() << " " << P1.z()
             << " " << col.x() << " " << col.y() << " " << col.z() << "\n";
        file << "v " << P2.x() << " " << P2.y() << " " << P2.z()
             << " " << col.x() << " " << col.y() << " " << col.z() << "\n";

        file << "l " << vertexCount + 1 << " " << vertexCount + 2 << "\n";
        vertexCount += 2;
    }

    // ============================================================
    // 3. KeyFrames + Colored camera axes + ID comment
    // ============================================================

    std::vector<Eigen::Vector3f> KF_COLORS = {
        {1,0,0},{0,1,0},{0,0,1},{1,1,0},{0,1,1},{1,0,1},{1,0.5,0},{0.6,0.3,1}
    };

    int kfi = 0;
    std::vector<int> KFcenterIndex; // 用来连轨迹

    for (auto pKF : keyframes) {
        if (!pKF) continue;

        int id = pKF->mnId;      // 假设存在
        Eigen::Vector3f col = KF_COLORS[kfi % KF_COLORS.size()];

        file << "\n# KeyFrame ID = " << id << "\n";  // OBJ comment

        Eigen::Vector3f C = pKF->GetCameraCenter();
        //Eigen::Matrix3f R = pKF->GetRcw();
        Sophus::SE3f Tcw = pKF->GetPose();
        Eigen::Matrix4f Tcw_eign = Tcw.matrix();
        Eigen::Matrix3f R = Tcw_eign.block<3,3>(0,0);

        Eigen::Vector3f X = C + axisLength * R.row(0).transpose();
        Eigen::Vector3f Y = C + axisLength * R.row(1).transpose();
        Eigen::Vector3f Z = C + axisLength * R.row(2).transpose();

        // Camera center
        file << "v " << C.x() << " " << C.y() << " " << C.z()
             << " " << col.x() << " " << col.y() << " " << col.z() << "\n";
        int camIdx = ++vertexCount;
        KFcenterIndex.push_back(camIdx);

        // axes
        file << "v " << X.x() << " " << X.y() << " " << X.z()
             << " 1 0 0\n";
        file << "l " << camIdx << " " << camIdx + 1 << "\n";
        vertexCount++;

        file << "v " << Y.x() << " " << Y.y() << " " << Y.z()
             << " 0 1 0\n";
        file << "l " << camIdx << " " << camIdx + 2 << "\n";
        vertexCount++;

        file << "v " << Z.x() << " " << Z.y() << " " << Z.z()
             << " 0 0 1\n";
        file << "l " << camIdx << " " << camIdx + 3 << "\n";
        vertexCount++;

        kfi++;
    }

    // ============================================================
    // 4. KeyFrame 轨迹 polyline
    // ============================================================

    file << "\n# KeyFrame trajectory polyline\n";
    for (size_t i = 1; i < KFcenterIndex.size(); i++) {
        file << "l " << KFcenterIndex[i-1] << " " << KFcenterIndex[i] << "\n";
    }

    file.close();
    std::cout << "[ExportFullSceneOBJ] Export successful to " << filename << "\n";
}


} // namespace ORB_SLAM3