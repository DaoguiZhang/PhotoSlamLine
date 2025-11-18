/**
* This file is part of Structure-SLAM.
* Copyright (C) 2025 zdg(daiguizhanghao@gmail.com) (zhejiang University of china)
*
*/
#include "LSDmatcher.h"
#include "Converter.h"
#include "LSDVisualizer.h"
#include <unordered_set>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

#define LSD_VISUALIZER_FLAG 1  // 1: enable visualization; 0: disable visualization

namespace ORB_SLAM3
{
    const int LSDmatcher::TH_HIGH = 60;
    const int LSDmatcher::TH_LOW = 50;

    void LSDmatcher::match(const std::vector<cv::line_descriptor::KeyLine>& keylines1, const cv::Mat& desc1,
               const std::vector<cv::line_descriptor::KeyLine>& keylines2, const cv::Mat& desc2,
               std::vector<cv::DMatch>& good_matches) 
    {
        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(desc1, desc2, knn_matches, 2);

        // ratio test
        std::vector<cv::DMatch> ratio_matches;
        for (size_t i = 0; i < knn_matches.size(); ++i) {
            if (knn_matches[i].size() < 2) continue;
            if (knn_matches[i][0].distance < mratio_thresh * knn_matches[i][1].distance)
                ratio_matches.push_back(knn_matches[i][0]);
        }

        // 角度和长度比过滤
        std::vector<cv::DMatch> geom_matches;
        for (auto &m : ratio_matches) {
            const cv::line_descriptor::KeyLine &kl1 = keylines1[m.queryIdx];
            const cv::line_descriptor::KeyLine &kl2 = keylines2[m.trainIdx];

            float len_ratio = kl1.lineLength / kl2.lineLength;
            if (len_ratio < 1.0f/mmax_length_ratio || len_ratio > mmax_length_ratio)
                continue;

            float angle_diff = std::fabs(kl1.angle - kl2.angle);
            angle_diff = std::fmod(angle_diff, 180.0f); // 保证在 0~180
            if (angle_diff > mmax_angle_diff)
                continue;

            geom_matches.push_back(m);
        }

        good_matches = geom_matches;

        // 几何验证：RANSAC 基于中心点
        if (good_matches.size() >= 4) {
            std::vector<cv::Point2f> pts1, pts2;
            for (auto &m : good_matches) {
                pts1.push_back(keylines1[m.queryIdx].pt);
                pts2.push_back(keylines2[m.trainIdx].pt);
            }

            cv::Mat inlierMask;
            cv::findHomography(pts1, pts2, RANSAC, mransac_threshold, inlierMask);

            std::vector<cv::DMatch> inlier_matches;
            for (size_t i = 0; i < good_matches.size(); ++i)
                if (inlierMask.at<uchar>(i))
                    inlier_matches.push_back(good_matches[i]);

            good_matches.swap(inlier_matches);
        }
    }

    // // Descriptor distance using Hamming distance (for binary descriptors)
    // int LSDmatcher::DescriptorDistance(const cv::Mat& desc1, const cv::Mat& desc2) {
    // // Ensure descriptors are of type CV_8U (binary)
    //     if (desc1.type() != CV_8U || desc2.type() != CV_8U) {
    //         std::cerr << "Error: Descriptors must be binary!" << std::endl;
    //         return -1;
    //     }
    //     // Use cv::norm with NORM_HAMMING for binary descriptors
    //     return cv::norm(desc1, desc2, cv::NORM_HAMMING);
    // }

    int LSDmatcher::DescriptorDistance(const cv::Mat& a, const cv::Mat& b)
    {
    // 基础检查（两种版本都适用）
    assert(a.cols == b.cols && a.rows == b.rows);
    assert(a.type() == CV_8U && b.type() == CV_8U);

    #ifdef USE_FAST_HAMMING
        // -------------------------
        // 🔥 高速版本（手写 XOR）
        // -------------------------
        const int* pa = a.ptr<int32_t>();
        const int* pb = b.ptr<int32_t>();
        int dist = 0;

        // 假设每个描述子为 256 bits（= 32 bytes = 8×int）
        for (int i = 0; i < 8; i++, pa++, pb++)
        {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }
        return dist;

#else
        // -------------------------
        // 🧩 安全版本（OpenCV 原生）
        // -------------------------
        return cv::norm(a, b, cv::NORM_HAMMING);
#endif
    }


    int LSDmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono) {
        int nmatches = 0;

        Eigen::Matrix4f Tcw_g = CurrentFrame.GetPose().matrix();
        const cv::Mat Tcw = Converter::toCvMat(Tcw_g);
        //const Eigen::Vector3f twc = Tcw.inverse().translation();

        const cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = Tcw.rowRange(0, 3).col(3);

        const cv::Mat twc = -Rcw.t()*tcw;

        Eigen::Matrix4f Tcmlw_g = LastFrame.GetPose().matrix();
        const cv::Mat Tcmlw = Converter::toCvMat(Tcmlw_g);
        const cv::Mat Rlw = Tcmlw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tlw = Tcmlw.rowRange(0, 3).col(3);

        const cv::Mat tlc = Rlw*twc+tlw;

        const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
        const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

        for (int i = 0; i < LastFrame.NL; i++) {
            MapLine *pML = LastFrame.mvpMapLines[i];

            if (!pML || pML->isBad() || LastFrame.mvbLineOutlier[i]) {
                continue;
            }

            //Vector6d P = pML->GetWorldPos();
            std::pair<Eigen::Vector3f, Eigen::Vector3f> P = pML->GetLineWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P.first(0), P.first(1), P.first(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P.second(3), P.second(4), P.second(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const auto &SPcX = SPc.at<float>(0);
            const auto &SPcY = SPc.at<float>(1);
            const auto &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const auto &EPcX = EPc.at<float>(0);
            const auto &EPcY = EPc.at<float>(1);
            const auto &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = CurrentFrame.fx * SPcX * invz1 + CurrentFrame.cx;
            const float v1 = CurrentFrame.fy * SPcY * invz1 + CurrentFrame.cy;

            if (u1 < CurrentFrame.mnMinX || u1 > CurrentFrame.mnMaxX)
                continue;
            if (v1 < CurrentFrame.mnMinY || v1 > CurrentFrame.mnMaxY)
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = CurrentFrame.fx * EPcX * invz2 + CurrentFrame.cx;
            const float v2 = CurrentFrame.fy * EPcY * invz2 + CurrentFrame.cy;

            if (u2 < CurrentFrame.mnMinX || u2 > CurrentFrame.mnMaxX)
                continue;
            if (v2 < CurrentFrame.mnMinY || v2 > CurrentFrame.mnMaxY)
                continue;

            int nLastOctave = LastFrame.mvKeys[i].octave;

            float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

            vector<size_t> vIndices;

            if(bForward)
            {
                vIndices = CurrentFrame.GetLinesInArea(u1, v1, u2, v2, radius, nLastOctave);
            }
            else if(bBackward)
            {
                vIndices = CurrentFrame.GetLinesInArea(u1, v1, u2, v2, radius, 0, nLastOctave);
            }
            else
            {   
                vIndices = CurrentFrame.GetLinesInArea(u1, v1, u2, v2, radius, nLastOctave-1, nLastOctave+1);
            }
                

            if(vIndices.empty())
                continue;

            const cv::Mat desc = pML->GetLineDescriptor();

            int bestDist=256;
            int bestLevel= -1;
            int bestDist2=256;
            int bestLevel2 = -1;
            int bestIdx =-1 ;

            for(unsigned long idx : vIndices)
            {
                if( CurrentFrame.mvpMapLines[idx])
                    if( CurrentFrame.mvpMapLines[idx]->Observations()>0)
                        continue;

                const cv::Mat &d =  CurrentFrame.mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(desc, d);

                if(dist<bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel =  CurrentFrame.mvKeyLinesUn[idx].octave;
                    bestIdx = idx;
                }
                else if(dist < bestDist2)
                {
                    bestLevel2 =  CurrentFrame.mvKeyLinesUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            if(bestDist <= TH_HIGH)
            {
                if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                    continue;

                CurrentFrame.mvpMapLines[bestIdx]=pML;
                nmatches++;
            }
        }

        return nmatches;
    }

    // //C++17 version, CHECK NEXT
    // int LSDmatcher::SearchByProjectionNew(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool isMono)
    // {
    //     int nMatches = 0;
    //     //=== 1. 相对位姿: 从 Last -> Current
    //     const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    //     const Sophus::SE3f Tlw = LastFrame.GetPose();
    //     // 从上一帧到当前帧的变换（last → current）
    //     const Sophus::SE3f Tcl = Tcw * Tlw.inverse();
    //     const Eigen::Matrix3f Rcl = Tcl.rotationMatrix();
    //     const Eigen::Vector3f tcl = Tcl.translation();
    //     const float fx = CurrentFrame.fx;
    //     const float fy = CurrentFrame.fy;
    //     const float cx = CurrentFrame.cx;
    //     const float cy = CurrentFrame.cy;
    //     const float mb = CurrentFrame.mb;
    //     const bool isForward = tcl(2) > mb && !isMono;
    //     const bool isBackward = tcl(2) < -mb && !isMono;
    //     const int nLastLines = LastFrame.NL;
    //     //const int TH_HIGH = 60;
    //     const float nn_ratio = 0.8f;
    //     std::cerr << "CurrentFrame.mnMinX, CurrentFrame.mnMaxX: " << CurrentFrame.mnMinX << ", " << CurrentFrame.mnMaxX << std::endl;
    //     std::cerr << "CurrentFrame.mnMinY, CurrentFrame.mnMaxY: " << CurrentFrame.mnMinY << ", " << CurrentFrame.mnMaxY << std::endl;
    //     //=== 2. 遍历上一帧线段
    //     for (int iL = 0; iL < nLastLines; iL++)
    //     {
    //         MapLine* pML = LastFrame.mvpMapLines[iL];
    //         if (!pML || pML->isBad() || LastFrame.mvbLineOutlier[iL])
    //         {
    //             // if(!pML)
    //             // {
    //             //     std::cerr << "iL:" << iL << ";    LSDmatcher->SearchByProjection pML is null!" << std::endl;
    //             //     continue;
    //             // }
    //             // if(pML->isBad())
    //             // {
    //             //     std::cerr << "iL:" << iL << ";    LSDmatcher->SearchByProjection pML is bad!" << std::endl;
    //             //     continue;
    //             // }
    //             // if(LastFrame.mvbLineOutlier[iL])
    //             // {
    //             //     std::cerr << "iL:" << iL << ";    LSDmatcher->SearchByProjection LastFrame.mvbLineOutlier[iL] is true!" << std::endl;
    //             //     continue;
    //             // }
    //             continue;
    //         }
    //         //=== 3. 获取线段3D端点（世界坐标）
    //         const auto line3D = pML->GetLineWorldPos();
    //         const Eigen::Vector3f P1w = line3D.first;
    //         const Eigen::Vector3f P2w = line3D.second;
    //         //=== 4. 投影到当前帧相机坐标系
    //         const Eigen::Vector3f P1c = Tcw * P1w;
    //         const Eigen::Vector3f P2c = Tcw * P2w;
    //         if (P1c(2) <= 0 || P2c(2) <= 0)
    //             continue;
    //         const float invz1 = 1.0f / P1c(2);
    //         const float invz2 = 1.0f / P2c(2);
    //         const float u1 = fx * P1c(0) * invz1 + cx;
    //         const float v1 = fy * P1c(1) * invz1 + cy;
    //         const float u2 = fx * P2c(0) * invz2 + cx;
    //         const float v2 = fy * P2c(1) * invz2 + cy;
    //         if (u1 < CurrentFrame.mnMinX || u1 > CurrentFrame.mnMaxX ||
    //             v1 < CurrentFrame.mnMinY || v1 > CurrentFrame.mnMaxY ||
    //             u2 < CurrentFrame.mnMinX || u2 > CurrentFrame.mnMaxX ||
    //             v2 < CurrentFrame.mnMinY || v2 > CurrentFrame.mnMaxY)
    //             continue;
    //         //=== 5. 搜索中心为中点
    //         const float um = 0.5f * (u1 + u2);
    //         const float vm = 0.5f * (v1 + v2);
    //         const int lastOctave = LastFrame.mvKeyLines[iL].octave;
    //         const float searchRadius = th * CurrentFrame.mvScaleFactors[lastOctave];
    //         std::cerr << "LSDmatcher->SearchByProjection um, vm, searchRadius: " << um << ", " << vm << ", " << searchRadius << std::endl;
    //         std::cerr << "LSDmatcher->SearchByProjection CurrentFrame.mvScaleFactors[lastOctave], th: " << CurrentFrame.mvScaleFactors[lastOctave] << ", " << th << std::endl;
    //         std::cerr << "LSDmatcher->SearchByProjection lastOctave, lastOctave: " << lastOctave << ", " << lastOctave << std::endl;
    //         std::vector<std::size_t> vIndices;
    //         if (isForward)
    //             vIndices = CurrentFrame.GetLinesInAreaMean(um, vm, searchRadius, lastOctave, lastOctave);
    //         else if (isBackward)
    //             vIndices = CurrentFrame.GetLinesInAreaMean(um, vm, searchRadius, 0, lastOctave);
    //         else
    //             vIndices = CurrentFrame.GetLinesInAreaMean(um, vm, searchRadius, lastOctave - 1, lastOctave + 1);
    //         if (vIndices.empty())
    //             continue;
    //         //=== 6. 方向一致性检测
    //         const cv::Point2f dirLast(u2 - u1, v2 - v1);
    //         const float normDirLast = cv::norm(dirLast);
    //         if (normDirLast < 5.0f) continue;
    //         const cv::Mat descLast = LastFrame.mLineDescriptors.row(iL);
    //         int bestDist = 256;
    //         int secondBestDist = 256;
    //         int bestIdx = -1;
    //         //=== 7. 遍历候选线段
    //         for (size_t j : vIndices)
    //         {
    //             if( CurrentFrame.mvpMapLines[j])
    //                 if( CurrentFrame.mvpMapLines[j]->Observations()>0)
    //                     continue;
    //             const cv::line_descriptor::KeyLine &kl = CurrentFrame.mvKeyLinesUn[j];
    //             const cv::Point2f dirCur(kl.endPointX - kl.startPointX, kl.endPointY - kl.startPointY);
    //             const float cosang = (dirLast.x * dirCur.x + dirLast.y * dirCur.y) /
    //                              (normDirLast * cv::norm(dirCur) + 1e-6);
    //             if (cosang < 0.9)
    //                 continue;
    //             if (CurrentFrame.mvpMapLines[j])
    //                 continue;
    //             const cv::Mat &descCur = CurrentFrame.mLineDescriptors.row(j);
    //             const int dist = cv::norm(descLast, descCur, cv::NORM_HAMMING);
    //             if (dist < bestDist)
    //             {
    //                 secondBestDist = bestDist;
    //                 bestDist = dist;
    //                 bestIdx = j;
    //             }
    //             else if (dist < secondBestDist)
    //             {
    //                 secondBestDist = dist;
    //             }
    //         }
    //         //=== 8. 最近邻判定
    //         if (bestDist < TH_HIGH && bestDist <= nn_ratio * secondBestDist)
    //         {
    //             CurrentFrame.mvpMapLines[bestIdx] = pML;
    //             CurrentFrame.mvbLineOutlier[bestIdx] = false;
    //             nMatches++;
    //         }
    //     }
    //     return nMatches;
    // }

    int LSDmatcher::SearchByProjectionNew(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool isMono)
    {
        int nMatches = 0;
        //=== 1. 获取位姿
        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        const Sophus::SE3f Tlw = LastFrame.GetPose();
        const Sophus::SE3f Tlc = Tcw.inverse() * Tlw; // Last -> Current
        const auto &LastLines = LastFrame.mvKeyLines;
        const auto &LastDescriptors = LastFrame.mLineDescriptors;
        const int NL = LastLines.size();
        std::vector<bool> vbAlreadyMatched(CurrentFrame.mvKeyLines.size(), false);
        //=== 2. 遍历上一帧所有线特征
        for (int i = 0; i < NL; i++)
        {
            MapLine* pML = LastFrame.mvpMapLines[i];
            if (!pML || pML->isBad())
                continue;
            // 2.1 投影端点到当前帧
            Eigen::Vector3f P1c = Tlc * pML->GetLineWorldPos().first;
            Eigen::Vector3f P2c = Tlc * pML->GetLineWorldPos().second;
            if (P1c[2] <= 0 || P2c[2] <= 0)
                continue; // 深度检查
            Eigen::Vector2f uv1 = CurrentFrame.mpCamera->project(P1c);
            Eigen::Vector2f uv2 = CurrentFrame.mpCamera->project(P2c);
            // 2.2 搜索矩形加大一定范围
            float minX = std::min(uv1[0], uv2[0]) - th;
            float maxX = std::max(uv1[0], uv2[0]) + th;
            float minY = std::min(uv1[1], uv2[1]) - th;
            float maxY = std::max(uv1[1], uv2[1]) + th;
            //=== 3. 匹配当前帧线
            int bestIdx = -1;
            float bestScore = std::numeric_limits<float>::max();
            for (int j = 0; j < CurrentFrame.mvKeyLines.size(); j++)
            {
                if (vbAlreadyMatched[j])
                    continue;
                const cv::line_descriptor::KeyLine &kl = CurrentFrame.mvKeyLines[j];
                // 3.1 端点在搜索区域内
                if (kl.startPointX < minX || kl.startPointX > maxX || kl.startPointY < minY || kl.startPointY > maxY)
                    continue;
                if (kl.endPointX < minX || kl.endPointX > maxX || kl.endPointY < minY || kl.endPointY > maxY)
                    continue;
                // 3.2 计算描述子距离
                float descDist = DescriptorDistance(LastDescriptors.row(i), CurrentFrame.mLineDescriptors.row(j));
                // 3.3 端点视差一致性约束（考虑方向不一致）
                float dx1 = kl.startPointX - uv1[0];
                float dy1 = kl.startPointY - uv1[1];
                float dx2 = kl.endPointX - uv2[0];
                float dy2 = kl.endPointY - uv2[1];
                float err_forward = std::sqrt(dx1*dx1 + dy1*dy1) + std::sqrt(dx2*dx2 + dy2*dy2);

                // 反向情况：当前帧线方向与上一帧相反
                float dx1r = kl.startPointX - uv2[0];
                float dy1r = kl.startPointY - uv2[1];
                float dx2r = kl.endPointX - uv1[0];
                float dy2r = kl.endPointY - uv1[1];
                float err_reverse = std::sqrt(dx1r*dx1r + dy1r*dy1r) + std::sqrt(dx2r*dx2r + dy2r*dy2r);

                // 取较小误差
                float endpointError = std::min(err_forward, err_reverse);

                // 3.4 综合评分（描述子距离 + 端点误差）
                float score = descDist + endpointError * 0.01f; // 权重可调
                if (score < bestScore)
                {
                    bestScore = score;
                    bestIdx = j;
                }
            }
            //=== 4. 匹配成功条件
            if (bestIdx >= 0 && bestScore <= th)
            {
                CurrentFrame.mvpMapLines[bestIdx] = pML;
                vbAlreadyMatched[bestIdx] = true;
                CurrentFrame.mvbLineOutlier[bestIdx] = false;
                nMatches++;
            }
        }
        return nMatches;
    }

    void LSDmatcher::DebugSearchByProjectionNew(
        Frame &CurrentFrame,
        const Frame &LastFrame,
        const std::string &windowName)
    {
        cv::Mat imgDraw = CurrentFrame.imgLeftRGB.clone();
        if (imgDraw.channels() == 1)
            cv::cvtColor(imgDraw, imgDraw, cv::COLOR_GRAY2BGR);
        //=== 获取位姿
        const Sophus::SE3f Tcw = CurrentFrame.GetPose();  // 当前帧
        const Sophus::SE3f Tlw = LastFrame.GetPose();     // 上一帧
        const Sophus::SE3f Tlc = Tcw.inverse() * Tlw;     // Last -> Current
        int nValidProj = 0;
        for (size_t i = 0; i < LastFrame.mvpMapLines.size(); ++i)
        {
            MapLine *pML = LastFrame.mvpMapLines[i];
            if (!pML || pML->isBad())
                continue;
            //=== 取世界坐标端点
            const auto &P1w = pML->GetLineWorldPos().first;
            const auto &P2w = pML->GetLineWorldPos().second;
            //=== 投影到当前帧相机坐标
            Eigen::Vector3f P1c = Tlc * P1w;
            Eigen::Vector3f P2c = Tlc * P2w;
            if (P1c[2] <= 0 || P2c[2] <= 0)
                continue;  // 深度无效
            //=== 投影到像素坐标
            Eigen::Vector2f uv1 = CurrentFrame.mpCamera->project(P1c);
            Eigen::Vector2f uv2 = CurrentFrame.mpCamera->project(P2c);
            //=== 检查是否在图像范围内
            if (uv1[0] < 0 || uv1[0] >= imgDraw.cols || uv1[1] < 0 || uv1[1] >= imgDraw.rows ||
                uv2[0] < 0 || uv2[0] >= imgDraw.cols || uv2[1] < 0 || uv2[1] >= imgDraw.rows)
                continue;
            //=== 绘制红线（投影线）
            cv::Point2f p1(uv1[0], uv1[1]);
            cv::Point2f p2(uv2[0], uv2[1]);
            cv::line(imgDraw, p1, p2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::circle(imgDraw, p1, 3, cv::Scalar(0, 255, 0), -1);
            cv::circle(imgDraw, p2, 3, cv::Scalar(0, 255, 0), -1);
            //=== 写编号
            cv::putText(imgDraw, std::to_string(i), p1, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255, 255, 255), 1);
            nValidProj++;
        }
        //=== 绘制当前帧检测到的线（蓝色，用于对比）
        for (size_t i = 0; i < CurrentFrame.mvKeyLines.size(); ++i)
        {
            const auto &kl = CurrentFrame.mvKeyLines[i];
            cv::Point2f sp(kl.startPointX, kl.startPointY);
            cv::Point2f ep(kl.endPointX, kl.endPointY);
            cv::line(imgDraw, sp, ep, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }
        std::cout << "[Debug] 投影到当前帧的有效 MapLines 数量: " 
                << nValidProj << " / " << LastFrame.mvpMapLines.size() << std::endl;

        cv::imshow(windowName, imgDraw);
        cv::waitKey(0);
    }

    void LSDmatcher::DebugDrawProjectedLineFrame(Frame &CurrentFrame, std::string &windowName)
    {
        cv::Mat imgDraw = CurrentFrame.imgLeftRGB.clone();
        if (imgDraw.channels() == 1)
            cv::cvtColor(imgDraw, imgDraw, cv::COLOR_GRAY2BGR);
        //=== 当前帧的相机位姿
        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        for (size_t i = 0; i < CurrentFrame.mvKeyLines.size(); ++i)
        {
            const auto &kl = CurrentFrame.mvKeyLines[i];
            cv::Point2f sp(kl.startPointX, kl.startPointY);
            cv::Point2f ep(kl.endPointX, kl.endPointY);
            //=== 蓝色线段：当前帧检测线
            cv::line(imgDraw, sp, ep, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            cv::putText(imgDraw, std::to_string(i), sp, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255, 255, 255), 1);
            //=== 匹配的地图线段
            MapLine* pML = CurrentFrame.mvpMapLines[i];
            if (pML && !pML->isBad())
            {
                // 取世界坐标的端点
                const Eigen::Vector3f &P1w = pML->GetLineWorldPos().first;
                const Eigen::Vector3f &P2w = pML->GetLineWorldPos().second;
                //=== 投影到当前帧
                Eigen::Vector3f P1c = Tcw * P1w;
                Eigen::Vector3f P2c = Tcw * P2w;
                // 跳过无效深度
                if (P1c[2] <= 0 || P2c[2] <= 0)
                    continue;

                Eigen::Vector2f uv1 = CurrentFrame.mpCamera->project(P1c);
                Eigen::Vector2f uv2 = CurrentFrame.mpCamera->project(P2c);
                cv::Point2f p1_proj(uv1[0], uv1[1]);
                cv::Point2f p2_proj(uv2[0], uv2[1]);

                //=== 红线：投影线
                cv::line(imgDraw, p1_proj, p2_proj, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

                //=== 绿色点：端点投影位置
                cv::circle(imgDraw, p1_proj, 3, cv::Scalar(0, 255, 0), -1);
                cv::circle(imgDraw, p2_proj, 3, cv::Scalar(0, 255, 0), -1);

                //=== 浅蓝线：连接检测线和投影线端点（可视化偏差）
                cv::line(imgDraw, sp, p1_proj, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
                cv::line(imgDraw, ep, p2_proj, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            }
        }

        cv::imshow(windowName, imgDraw);
        cv::waitKey(0);
    }

    void LSDmatcher::DebugDrawLineMatches(const Frame &lastFrame, const Frame &currentFrame)
    {
        // 转灰度图（若原始是float图或三通道）
        cv::Mat img1, img2;
        if (lastFrame.imgLeftRGB.channels() == 3)
            lastFrame.imgLeftRGB.convertTo(img1, CV_8UC3, 255.0);
        else
            cv::cvtColor(lastFrame.imgLeftRGB, img1, cv::COLOR_GRAY2BGR);
        if (currentFrame.imgLeftRGB.channels() == 3)
            currentFrame.imgLeftRGB.convertTo(img2, CV_8UC3, 255.0);
        else
            cv::cvtColor(currentFrame.imgLeftRGB, img2, cv::COLOR_GRAY2BGR);
        // 拼接两张图
        const int rows = std::max(img1.rows, img2.rows);
        const int cols = img1.cols + img2.cols;
        cv::Mat outImg(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
        img1.copyTo(outImg(cv::Rect(0, 0, img1.cols, img1.rows)));
        img2.copyTo(outImg(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
        const int offsetX = img1.cols;
        // 绘制匹配线段
        for (size_t i = 0; i < currentFrame.mvpMapLines.size(); i++)
        {
            MapLine *pML = currentFrame.mvpMapLines[i];
            if (!pML || pML->isBad())
                continue;
            // 在上一帧中找到对应线段
            bool found = false;
            cv::line_descriptor::KeyLine kl1, kl2;
            for (size_t j = 0; j < lastFrame.mvpMapLines.size(); j++)
            {
                if (lastFrame.mvpMapLines[j] == pML)
                {
                    kl1 = lastFrame.mvKeyLinesUn[j];
                    kl2 = currentFrame.mvKeyLinesUn[i];
                    found = true;
                    break;
                }
            }
            if (!found)
                continue;
            // 上一帧线段端点
            cv::Point2f p1s(kl1.startPointX, kl1.startPointY);
            cv::Point2f p1e(kl1.endPointX, kl1.endPointY);

            // 当前帧线段端点（需要加上拼接偏移）
            cv::Point2f p2s(kl2.startPointX + offsetX, kl2.startPointY);
            cv::Point2f p2e(kl2.endPointX + offsetX, kl2.endPointY);

            // 绘制线段
            cv::Scalar color(0, 255, 0); // 绿色表示匹配成功
            cv::line(outImg, p1s, p1e, color, 2);
            cv::line(outImg, p2s, p2e, color, 2);

            // 绘制连线（两帧间的匹配连线）
            cv::Point2f mid1((p1s.x + p1e.x) / 2, (p1s.y + p1e.y) / 2);
            cv::Point2f mid2((p2s.x + p2e.x) / 2, (p2s.y + p2e.y) / 2);
            cv::line(outImg, mid1, mid2, cv::Scalar(255, 0, 0), 1); // 蓝色连接线
        }

        // 显示结果
        cv::imshow("Line Matches", outImg);
        cv::waitKey(0);
    }

    // int LSDmatcher::MatchLinesByProjection(Frame &currentFrame, const Frame &lastFrame, const float threshold, const bool isMono) {
    //     int numMatches = 0;
    //     Eigen::Matrix4f currentFramePose = currentFrame.GetPose().matrix();
    //     const cv::Mat currentFramePoseMat = Converter::toCvMat(currentFramePose);
    //     const cv::Mat rotationMatrixCurrent = currentFramePoseMat.rowRange(0, 3).colRange(0, 3);
    //     const cv::Mat translationVectorCurrent = currentFramePoseMat.rowRange(0, 3).col(3);
    //     const cv::Mat cameraToWorld = -rotationMatrixCurrent.t() * translationVectorCurrent;
    //     Eigen::Matrix4f lastFramePose = lastFrame.GetPose().matrix();
    //     const cv::Mat lastFramePoseMat = Converter::toCvMat(lastFramePose);
    //     const cv::Mat rotationMatrixLast = lastFramePoseMat.rowRange(0, 3).colRange(0, 3);
    //     const cv::Mat translationVectorLast = lastFramePoseMat.rowRange(0, 3).col(3);
    //     const cv::Mat cameraToLastFrame = rotationMatrixLast * cameraToWorld + translationVectorLast;
    //     const bool isForwardMatch = cameraToLastFrame.at<float>(2) > currentFrame.mb && !isMono;
    //     const bool isBackwardMatch = -cameraToLastFrame.at<float>(2) > currentFrame.mb && !isMono;
    //     for (int i = 0; i < lastFrame.NL; i++) {
    //         MapLine *line = lastFrame.mvpMapLines[i];
    //         if (!line || line->isBad() || lastFrame.mvbLineOutlier[i]) {
    //             continue;
    //         }
    //         std::pair<Eigen::Vector3f, Eigen::Vector3f> lineWorldPosition = line->GetLineWorldPos();
    //         cv::Mat startPointWorld = (cv::Mat_<float>(3, 1) << lineWorldPosition.first(0), lineWorldPosition.first(1), lineWorldPosition.first(2));
    //         cv::Mat endPointWorld = (cv::Mat_<float>(3, 1) << lineWorldPosition.second(0), lineWorldPosition.second(1), lineWorldPosition.second(2));
    //         const cv::Mat startPointCamera = rotationMatrixCurrent * startPointWorld + translationVectorCurrent;
    //         const auto &startPointCameraX = startPointCamera.at<float>(0);
    //         const auto &startPointCameraY = startPointCamera.at<float>(1);
    //         const auto &startPointCameraZ = startPointCamera.at<float>(2);
    //         const cv::Mat endPointCamera = rotationMatrixCurrent * endPointWorld + translationVectorCurrent;
    //         const auto &endPointCameraX = endPointCamera.at<float>(0);
    //         const auto &endPointCameraY = endPointCamera.at<float>(1);
    //         const auto &endPointCameraZ = endPointCamera.at<float>(2);
    //         if (startPointCameraZ < 0.0f || endPointCameraZ < 0.0f)
    //             continue;
    //         const float inverseStartPointZ = 1.0f / startPointCameraZ;
    //         const float u1 = currentFrame.fx * startPointCameraX * inverseStartPointZ + currentFrame.cx;
    //         const float v1 = currentFrame.fy * startPointCameraY * inverseStartPointZ + currentFrame.cy;
    //         if (u1 < currentFrame.mnMinX || u1 > currentFrame.mnMaxX)
    //             continue;
    //         if (v1 < currentFrame.mnMinY || v1 > currentFrame.mnMaxY)
    //             continue;
    //         const float inverseEndPointZ = 1.0f / endPointCameraZ;
    //         const float u2 = currentFrame.fx * endPointCameraX * inverseEndPointZ + currentFrame.cx;
    //         const float v2 = currentFrame.fy * endPointCameraY * inverseEndPointZ + currentFrame.cy;
    //         if (u2 < currentFrame.mnMinX || u2 > currentFrame.mnMaxX)
    //             continue;
    //         if (v2 < currentFrame.mnMinY || v2 > currentFrame.mnMaxY)
    //             continue;
    //         int lastFrameOctave = lastFrame.mvKeyLines[i].octave;
    //         float searchRadius = threshold * currentFrame.mvScaleFactors[lastFrameOctave];
    //         std::vector<size_t> candidateLineIndices;
    //         if (isForwardMatch)
    //             candidateLineIndices = currentFrame.GetLinesInArea(u1, v1, u2, v2, searchRadius, lastFrameOctave);
    //         else if (isBackwardMatch)
    //             candidateLineIndices = currentFrame.GetLinesInArea(u1, v1, u2, v2, searchRadius, 0, lastFrameOctave);
    //         else
    //             candidateLineIndices = currentFrame.GetLinesInArea(u1, v1, u2, v2, searchRadius, lastFrameOctave - 1, lastFrameOctave + 1);
    //         if (candidateLineIndices.empty())
    //             continue;
    //         const cv::Mat lineDescriptor = line->GetLineDescriptor();
    //         int bestMatchDistance = 256;
    //         int bestMatchLevel = -1;
    //         int secondBestMatchDistance = 256;
    //         int secondBestMatchLevel = -1;
    //         int bestMatchedIndex = -1;
    //         for (unsigned long index : candidateLineIndices) {
    //             if (currentFrame.mvpMapLines[index]) {
    //                 if (currentFrame.mvpMapLines[index]->Observations() > 0)
    //                     continue;
    //             }
    //             const cv::Mat &descriptor = currentFrame.mLineDescriptors.row(index);
    //             const int dist = DescriptorDistance(lineDescriptor, descriptor);
    //             if (dist < bestMatchDistance) {
    //                 secondBestMatchDistance = bestMatchDistance;
    //                 bestMatchDistance = dist;
    //                 secondBestMatchLevel = bestMatchLevel;
    //                 bestMatchLevel = currentFrame.mvKeyLinesUn[index].octave;
    //                 bestMatchedIndex = index;
    //             } else if (dist < secondBestMatchDistance) {
    //                 secondBestMatchLevel = currentFrame.mvKeyLinesUn[index].octave;
    //                 secondBestMatchDistance = dist;
    //             }
    //         }
    //         if (bestMatchDistance <= TH_HIGH) {
    //             if (bestMatchLevel == secondBestMatchLevel && bestMatchDistance > mfNNratio * secondBestMatchDistance)
    //                 continue;
    //             currentFrame.mvpMapLines[bestMatchedIndex] = line;
    //             numMatches++;
    //         }
    //     }
    //     return numMatches;
    // }


    int LSDmatcher::SerachForInitializeCV(Frame &InitialFrame, Frame &CurrentFrame, std::vector<std::pair<int, int>> &LineMatches)
    {
        LineMatches.clear();
        int nmatches = 0;
        const std::vector<cv::line_descriptor::KeyLine> &keylines1 = InitialFrame.mvKeyLinesUn;
        const cv::Mat &desc1 = InitialFrame.mLineDescriptors;
        const std::vector<cv::line_descriptor::KeyLine> &keylines2 = CurrentFrame.mvKeyLinesUn;
        const cv::Mat &desc2 = CurrentFrame.mLineDescriptors;
        std::vector<cv::DMatch> good_matches;
        match(keylines1, desc1, keylines2, desc2, good_matches);
        // store matches
        for (auto &m : good_matches) {
            LineMatches.emplace_back(m.queryIdx, m.trainIdx);
            nmatches++;
        }

#if LSD_VISUALIZER_FLAG   //visualization
        cv::Mat imgMatches = LSDVisualizer::DrawLineMatches(InitialFrame.imgLeftRGB, keylines1,
                                                           CurrentFrame.imgLeftRGB, keylines2,
                                                           good_matches);
        cv::imshow("Line Matches for Initialization", imgMatches);
        cv::waitKey(0);

#endif


        return nmatches;
    }
    
    // int LSDmatcher::SearchByProjection(KeyFrame* pKF,Frame &currentF, vector<MapLine*> &vpMapLineMatches)
    // {
    //     const std::vector<MapLine*> vpMapLinesKF = pKF->GetMapLineMatches();
    //     vpMapLineMatches = std::vector<MapLine*>(currentF.NL,static_cast<MapLine*>(NULL));
    //     int nmatches = 0;
    //     if(currentF.NL==0 || pKF->NL==0)
    //         return 0;
    //     cv::BFMatcher bfm(cv::NORM_HAMMING, false);
    //     cv::Mat ldesc1, ldesc2;
    //     std::vector<std::vector<cv::DMatch>> lmatches;
    //     ldesc1 = pKF->mLineDescriptors;
    //     ldesc2 = currentF.mLineDescriptors;
    //     //bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);
    //     bfm.knnMatch(ldesc1, ldesc2, lmatches, 2);
    //     if (lmatches.empty())
    //         return 0;
    //     double nn_dist_th, nn12_dist_th;
    //     const float minRatio=1.0f/1.5f;
    //     currentF.lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
    //     nn12_dist_th = nn12_dist_th*0.5;
    //     std::sort(lmatches.begin(), lmatches.end(), LSDmatcher::sortByQueryIdx);
    //     for(int i=0; i<lmatches.size(); i++)
    //     {
    //         //lmatches里装的 是匹配的编号，
    //         int qdx = lmatches[i][0].queryIdx;
    //         int tdx = lmatches[i][0].trainIdx;
    //         double dist_12 = lmatches[i][0].distance/lmatches[i][1].distance;
    //         //if(dist_12>nn12_dist_th)
    //         if(dist_12<minRatio)
    //         {
    //             MapLine* mapLine = vpMapLinesKF[qdx];
    //             if(mapLine)
    //             {
    //                 //cout<<"qdx and tdx"<<qdx<<","<<tdx<<endl;
    //                 vpMapLineMatches[tdx]=mapLine;
    //                 nmatches++;
    //             }
    //         }
    //     }
    //     return nmatches;
    // }

    int LSDmatcher::SearchByProjection(KeyFrame* pKF, Frame &currentF, vector<MapLine*> &vpMapLineMatches)
    {
        if(currentF.NL == 0 || pKF->NL == 0)
            return 0;

        const auto& vpMapLinesKF = pKF->GetMapLineMatches();
        vpMapLineMatches.assign(currentF.NL, nullptr);

        // 1. 描述子匹配
        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        std::vector<std::vector<cv::DMatch>> lmatches;
        matcher.knnMatch(pKF->mLineDescriptors, currentF.mLineDescriptors, lmatches, 2);
        if(lmatches.empty())
            return 0;
        // 2. Ratio test
        const float minRatio = 1.0f / 1.5f;
        std::sort(lmatches.begin(), lmatches.end(), LSDmatcher::sortByQueryIdx);
        int nmatches = 0;
        for(const auto& matchPair : lmatches)
        {
            const cv::DMatch& m0 = matchPair[0];
            const cv::DMatch& m1 = matchPair[1];
            if(m0.distance / m1.distance >= minRatio)
                continue;
            MapLine* mapLine = vpMapLinesKF[m0.queryIdx];
            if(mapLine)
            {
                vpMapLineMatches[m0.trainIdx] = mapLine;
                nmatches++;
            }
        }
        return nmatches;
    }


    void LSDmatcher::DebugDrawLineMatchesKeyFrame(KeyFrame* pKF, const Frame &currentFrame)
    {
        if (!pKF)
            return;
        //=== 1. 转灰度图（若原始是 float 图或三通道）
        cv::Mat img1, img2;
        if (pKF->imgLeftRGB.channels() == 3)
            pKF->imgLeftRGB.convertTo(img1, CV_8UC3, 255.0);
        else
            cv::cvtColor(pKF->imgLeftRGB, img1, cv::COLOR_GRAY2BGR);
        if (currentFrame.imgLeftRGB.channels() == 3)
            currentFrame.imgLeftRGB.convertTo(img2, CV_8UC3, 255.0);
        else
            cv::cvtColor(currentFrame.imgLeftRGB, img2, cv::COLOR_GRAY2BGR);
        //=== 2. 拼接两张图
        const int rows = std::max(img1.rows, img2.rows);
        const int cols = img1.cols + img2.cols;
        cv::Mat outImg(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
        img1.copyTo(outImg(cv::Rect(0, 0, img1.cols, img1.rows)));
        img2.copyTo(outImg(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
        const int offsetX = img1.cols;
        //=== 3. 绘制匹配线段
        for (size_t i = 0; i < currentFrame.mvpMapLines.size(); i++)
        {
            MapLine *pML = currentFrame.mvpMapLines[i];
            if (!pML || pML->isBad())
                continue;
            // 在关键帧中找到对应线段
            bool found = false;
            cv::line_descriptor::KeyLine kl1, kl2;
            for (size_t j = 0; j < pKF->GetMapLineMatches().size(); j++)
            {
                MapLine* pMLKF = pKF->GetMapLineMatches()[j];
                if (pMLKF == pML)
                {
                    kl1 = pKF->mvKeyLinesUn[j];
                    kl2 = currentFrame.mvKeyLinesUn[i];
                    found = true;
                    break;
                }
            }
            if (!found)
                continue;
            //=== 4. 上一关键帧线段端点
            cv::Point2f p1s(kl1.startPointX, kl1.startPointY);
            cv::Point2f p1e(kl1.endPointX, kl1.endPointY);
            //=== 5. 当前帧线段端点（加上拼接偏移）
            cv::Point2f p2s(kl2.startPointX + offsetX, kl2.startPointY);
            cv::Point2f p2e(kl2.endPointX + offsetX, kl2.endPointY);
            //=== 6. 绘制线段
            cv::Scalar color(0, 255, 0); // 绿色表示匹配成功
            cv::line(outImg, p1s, p1e, color, 2);
            cv::line(outImg, p2s, p2e, color, 2);
            //=== 7. 绘制连线（两帧间的匹配连线）
            cv::Point2f mid1((p1s.x + p1e.x) / 2, (p1s.y + p1e.y) / 2);
            cv::Point2f mid2((p2s.x + p2e.x) / 2, (p2s.y + p2e.y) / 2);
            cv::line(outImg, mid1, mid2, cv::Scalar(255, 0, 0), 1); // 蓝色连接线
        }
        //=== 8. 显示结果
        cv::imshow("Line Matches", outImg);
        cv::waitKey(0);
    }

    void LSDmatcher::DebugDrawLineMatchesFrame(Frame &CurrentFrame, std::string &windowName)
    {
        cv::Mat imgDraw = CurrentFrame.imgLeftRGB.clone();
        if (imgDraw.channels() == 1)
            cv::cvtColor(imgDraw, imgDraw, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < CurrentFrame.mvKeyLines.size(); ++i)
        {
            const auto &kl = CurrentFrame.mvKeyLines[i];
            cv::Point2f sp(kl.startPointX, kl.startPointY);
            cv::Point2f ep(kl.endPointX, kl.endPointY);
            //=== 蓝色线段（当前帧检测线）
            cv::line(imgDraw, sp, ep, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            cv::putText(imgDraw, std::to_string(i), sp, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255, 255, 255), 1);
            //=== 如果有对应 MapLine，画红色投影线段
            MapLine* pML = CurrentFrame.mvpMapLines[i];
            if (pML && !pML->isBad())
            {
                cv::Point2f p1_proj(pML->mLsTrackProjX, pML->mLsTrackProjY);
                cv::Point2f p2_proj(pML->mLeTrackProjX, pML->mLeTrackProjY);
                // 红线：地图线投影位置
                cv::line(imgDraw, p1_proj, p2_proj, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                // 绿色点：端点投影位置
                cv::circle(imgDraw, p1_proj, 3, cv::Scalar(0, 255, 0), -1);
                cv::circle(imgDraw, p2_proj, 3, cv::Scalar(0, 255, 0), -1);
                // 浅蓝线：连接检测线和投影线端点
                cv::line(imgDraw, sp, p1_proj, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
                cv::line(imgDraw, ep, p2_proj, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            }
        }
        cv::imshow(windowName, imgDraw);
        cv::waitKey(0);
    }

    // int LSDmatcher::SearchByProjection(Frame &F, const std::vector<MapLine *> &vpMapLines, const float th)
    // {
    //     int nmatches = 0;
    //     const bool bFactor = th!=1.0;
    //     for(auto pML : vpMapLines)
    //     {
    //         if(!pML || pML->isBad() || !pML->mbLineTrackInView)
    //             continue;
    //         const int &nPredictLevel = pML->mnLineTrackScaleLevel;
    //         float r = RadiusByViewingCos(pML->mLineTrackViewCos);
    //         if(bFactor)
    //             r*=th;
    //         vector<size_t> vIndices =
    //                 F.GetLinesInArea(pML->mLsTrackProjX, pML->mLsTrackProjY, pML->mLeTrackProjX, pML->mLeTrackProjY,
    //                                  r*F.mvScaleFactors[nPredictLevel], nPredictLevel-1, nPredictLevel);
    //         if(vIndices.empty())
    //             continue;
    //         const cv::Mat MLdescriptor = pML->GetLineDescriptor();
    //         int bestDist=256;
    //         int bestLevel= -1;
    //         int bestDist2=256;
    //         int bestLevel2 = -1;
    //         int bestIdx =-1 ;
    //         for(unsigned long idx : vIndices)
    //         {
    //             if(F.mvpMapLines[idx])
    //                 if(F.mvpMapLines[idx]->Observations()>0)
    //                     continue;
    //             const cv::Mat &d = F.mLineDescriptors.row(idx);
    //             const int dist = DescriptorDistance(MLdescriptor, d);
    //             // 根据描述子寻找描述子距离最小和次小的特征点
    //             if(dist<bestDist)
    //             {
    //                 bestDist2 = bestDist;
    //                 bestDist = dist;
    //                 bestLevel2 = bestLevel;
    //                 bestLevel = F.mvKeyLinesUn[idx].octave;
    //                 bestIdx = idx;
    //             }
    //             else if(dist < bestDist2)
    //             {
    //                 bestLevel2 = F.mvKeyLinesUn[idx].octave;
    //                 bestDist2 = dist;
    //             }
    //         }
    //         // Apply ratio to second match (only if best and second are in the same scale level)
    //         if(bestDist <= TH_HIGH)
    //         {
    //             if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
    //                 continue;
    //             F.mvpMapLines[bestIdx]=pML;
    //             nmatches++;
    //         }
    //     }
    //     return nmatches;
    // }

    int LSDmatcher::SearchByProjection(Frame &F, const std::vector<MapLine *> &vpMapLines, const float th)
    {
        int nmatches = 0;
        const bool bFactor = (th != 1.0f);

        for (auto pML : vpMapLines)
        {
            if (!pML || pML->isBad() || !pML->mbLineTrackInView)
                continue;
            //std::cerr << "LSDmatcher::SearchByProjection processing MapLine id: " << pML->mnId << std::endl;
            //std::cerr << "  TrackProjStart: (" << pML->mLsTrackProjX << ", " << pML->mLsTrackProjY << ")" << std::endl;
            //std::cerr << "  TrackProjEnd:   (" << pML->mLeTrackProjX << ", " << pML->mLeTrackProjY << ")" << std::endl;
            //std::cerr << "  LineTrackViewCos: " << pML->mLineTrackViewCos << std::endl;
            //std::cerr << "  LineTrackScaleLevel: " << pML->mnLineTrackScaleLevel << std::endl;
            const int nPredictLevel = pML->mnLineTrackScaleLevel;
            float r = RadiusByViewingCos(pML->mLineTrackViewCos);

            if (bFactor)
                r *= th;

            // 计算线段中心投影位置
            const float u_mean = 0.5f * (pML->mLsTrackProjX + pML->mLeTrackProjX);
            const float v_mean = 0.5f * (pML->mLsTrackProjY + pML->mLeTrackProjY);

            // 保护边界：确保金字塔层级不越界
            const int minLevel = std::max(0, nPredictLevel - 1);
            const int maxLevel = std::min(nPredictLevel + 1, F.mnScaleLevels - 1);
            //std::cerr << "  Search Area Center: (" << u_mean << ", " << v_mean << "), Radius: " << r * F.mvScaleFactors[nPredictLevel] << std::endl;

            std::vector<size_t> vIndices =
                F.GetLinesInAreaMean(u_mean, v_mean, r * F.mvScaleFactors[nPredictLevel]*4,
                                minLevel, maxLevel);

            if (vIndices.empty())
                continue;

            const cv::Mat MLdescriptor = pML->GetLineDescriptor();

            int bestDist = 256;
            int bestDist2 = 256;
            int bestIdx = -1;
            int bestLevel = -1;
            int bestLevel2 = -1;

            for (size_t idx : vIndices)
            {
                if (idx >= F.mvKeyLinesUn.size()) // 防越界
                    continue;

                if (F.mvpMapLines[idx] && F.mvpMapLines[idx]->Observations() > 0)
                    continue;

                const cv::Mat &d = F.mLineDescriptors.row(idx);
                const int dist = DescriptorDistance(MLdescriptor, d);
                if (dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestLevel2 = bestLevel;
                    bestDist = dist;
                    bestIdx = idx;
                    bestLevel = F.mvKeyLinesUn[idx].octave;
                }
                else if (dist < bestDist2)
                {
                    bestDist2 = dist;
                    bestLevel2 = F.mvKeyLinesUn[idx].octave;
                }
            }
            // ====== 过滤劣质匹配 ======
            const int TH_HIGH_NEW = 100;   // 建议阈值范围 [80,120]
            const float NN_RATIO = mfNNratio; // 通常0.8
            if (bestDist > TH_HIGH_NEW)
                continue;
            if (bestLevel == bestLevel2 && bestDist > NN_RATIO * bestDist2)
                continue;
            // ====== 成功匹配 ======
            F.mvpMapLines[bestIdx] = pML;
            pML->IncreaseFound();
            nmatches++;
        }
        return nmatches;
    }

    void LSDmatcher::DebugSearchByProjectionLinesMatch(
        Frame &F,
        std::vector<MapLine*> &vpMapLines,
        const std::string &windowName)
    {
        cv::Mat imgDraw = F.imgLeftRGB.clone();
        if (imgDraw.channels() == 1)
            cv::cvtColor(imgDraw, imgDraw, cv::COLOR_GRAY2BGR);
        const Sophus::SE3f Tcw = F.GetPose();  // 世界 -> 当前帧
        int nProj = 0, nMatched = 0;
        for (size_t iML = 0; iML < vpMapLines.size(); ++iML)
        {
            MapLine *pML = vpMapLines[iML];
            if (!pML || pML->isBad() || !pML->mbLineTrackInView)
                continue;
            //=== 世界坐标 -> 相机坐标（线段两个端点）
            Eigen::Vector3f P1w = pML->GetLineWorldPos().first;;
            Eigen::Vector3f P2w = pML->GetLineWorldPos().second;
            Eigen::Vector3f P1c = Tcw * P1w;
            Eigen::Vector3f P2c = Tcw * P2w;
            //=== 过滤无效深度
            if (P1c[2] <= 0 || P2c[2] <= 0)
                continue;
            //=== 投影到像素坐标
            Eigen::Vector2f uv1 = F.mpCamera->project(P1c);
            Eigen::Vector2f uv2 = F.mpCamera->project(P2c);
            //=== 过滤越界
            if ((uv1[0] < 0 || uv1[0] >= imgDraw.cols || uv1[1] < 0 || uv1[1] >= imgDraw.rows) ||
                (uv2[0] < 0 || uv2[0] >= imgDraw.cols || uv2[1] < 0 || uv2[1] >= imgDraw.rows))
                continue;

            nProj++;
            cv::Point2f p1(uv1[0], uv1[1]);
            cv::Point2f p2(uv2[0], uv2[1]);
            //=== 绘制预测线段（红色）
            cv::line(imgDraw, p1, p2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

            //=== 查找匹配
            bool bMatched = false;
            int matchedIdx = -1;
            for (size_t i = 0; i < F.mvpMapLines.size(); ++i)
            {
                if (F.mvpMapLines[i] == pML)
                {
                    bMatched = true;
                    matchedIdx = i;
                    break;
                }
            }
            if (bMatched)
            {
                nMatched++;
                const KeyLine &kl = F.mvKeyLines[matchedIdx];
                cv::Point2f kp1 = kl.getStartPoint();
                cv::Point2f kp2 = kl.getEndPoint();
                //=== 绘制匹配到的线段（绿色）
                cv::line(imgDraw, kp1, kp2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                //=== 连线连接预测与匹配（细线）
                cv::line(imgDraw, (p1 + p2) * 0.5f, (kp1 + kp2) * 0.5f,
                        cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                //=== 标号
                cv::putText(imgDraw, std::to_string(iML),
                            (p1 + p2) * 0.5f + cv::Point2f(3, -3),
                            cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255, 255, 255), 1);
            }
            else
            {
                // 未匹配到
                cv::putText(imgDraw, "X",
                            (p1 + p2) * 0.5f + cv::Point2f(3, -3),
                            cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 0, 255), 1);
            }
        }
        std::cout << "[Debug-Line] 投影线段数: " << nProj
                << " | 成功匹配: " << nMatched
                << " | 失败: " << (nProj - nMatched) << std::endl;

        cv::imshow(windowName, imgDraw);
        cv::waitKey(0);
    }


    //debug SearchByDescriptor(Frame &F, const std::vector<MapLine *> &vpMapLines, const float th)
    void LSDmatcher::DebugLineMatchSearchbyProjection(Frame &F, const std::vector<MapLine *> &vpMapLines, const float th)
    {
        cv::Mat imDebug;
        
        F.imgLeftRGB.copyTo(imDebug);

        // 绘制投影的 MapLine
        for (auto pML : vpMapLines)
        {
            if(!pML || pML->isBad() || !pML->mbLineTrackInView)
                continue;
            cv::Point2f p1(pML->mLsTrackProjX, pML->mLsTrackProjY);
            cv::Point2f p2(pML->mLeTrackProjX, pML->mLeTrackProjY);
            cv::line(imDebug, p1, p2, cv::Scalar(0,255,0), 2);  // 绿色线段表示MapLine投影
        }
        // 绘制当前帧检测到的线段
        for (size_t i=0; i<F.mvKeyLinesUn.size(); i++)
        {
            const cv::line_descriptor::KeyLine &kl = F.mvKeyLinesUn[i];
            cv::line(imDebug, cv::Point2f(kl.startPointX, kl.startPointY),
                        cv::Point2f(kl.endPointX, kl.endPointY),
                        cv::Scalar(255,0,0), 1); // 蓝色表示检测线段
        }
        // 绘制匹配成功的线段
        for (size_t i=0; i<F.mvpMapLines.size(); i++)
        {
            if(F.mvpMapLines[i])
            {
                const cv::line_descriptor::KeyLine &kl = F.mvKeyLinesUn[i];
                cv::line(imDebug, cv::Point2f(kl.startPointX, kl.startPointY),
                            cv::Point2f(kl.endPointX, kl.endPointY),
                            cv::Scalar(0,0,255), 2); // 红色表示匹配成功的线段
            }
        }
        // 显示窗口
        cv::imshow("Line Projection Debug", imDebug);
        cv::waitKey(0);
    }

    void LSDmatcher::DebugLineProjectionNew(Frame &F, const std::vector<MapLine*> &vpMapLines, const std::string &winName)
    {
        // 拷贝彩色图像用于可视化
        cv::Mat imDebug;
        F.imgLeftRGB.copyTo(imDebug);

        // 颜色定义
        cv::Scalar colorDetected(255, 0, 0);   // 蓝色：检测到的线段
        cv::Scalar colorProjected(0, 255, 0);  // 绿色：反投影线段
        cv::Scalar colorLink(0, 255, 255);     // 黄色：连接线
        cv::Scalar colorMatch(0, 0, 255);      // 红色：匹配成功
        // ========== [1] 绘制当前帧检测的线段 ==========
        for (size_t i = 0; i < F.mvKeyLinesUn.size(); i++)
        {
            const auto &kl = F.mvKeyLinesUn[i];
            cv::line(imDebug,
                    cv::Point2f(kl.startPointX, kl.startPointY),
                    cv::Point2f(kl.endPointX, kl.endPointY),
                    colorDetected, 1);
        }
        // ========== [2] 绘制 MapLine 的反投影线段 ==========
        for (auto pML : vpMapLines)
        {
            if (!pML || pML->isBad() || !pML->mbLineTrackInView)
                continue;

            cv::Point2f p1(pML->mLsTrackProjX, pML->mLsTrackProjY);
            cv::Point2f p2(pML->mLeTrackProjX, pML->mLeTrackProjY);

            // 绘制绿色反投影线
            cv::line(imDebug, p1, p2, colorProjected, 2);
            // 绘制线段中点
            cv::circle(imDebug, (p1 + p2) * 0.5f, 3, colorProjected, -1);
            // 可选：显示线段ID
            cv::putText(imDebug, std::to_string(pML->mnId),
                        (p1 + p2) * 0.5f + cv::Point2f(5, 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, colorProjected, 1);
        }

        // ========== [3] 绘制匹配成功的线段 ==========
        for (size_t i = 0; i < F.mvpMapLines.size(); i++)
        {
            if (F.mvpMapLines[i])
            {
                const auto &kl = F.mvKeyLinesUn[i];
                cv::line(imDebug,
                        cv::Point2f(kl.startPointX, kl.startPointY),
                        cv::Point2f(kl.endPointX, kl.endPointY),
                        colorMatch, 2);

                // 在匹配线段与其投影之间画虚线连接
                auto pML = F.mvpMapLines[i];
                cv::Point2f projCenter((pML->mLsTrackProjX + pML->mLeTrackProjX) / 2,
                                   (pML->mLsTrackProjY + pML->mLeTrackProjY) / 2);
                cv::Point2f detCenter((kl.startPointX + kl.endPointX) / 2,
                                  (kl.startPointY + kl.endPointY) / 2);
                cv::line(imDebug, projCenter, detCenter, colorLink, 1, cv::LINE_AA);
            }
        }

        // ========== [4] 输出投影调试信息 ==========
        for (auto pML : vpMapLines)
        {
            if (!pML || pML->isBad() || !pML->mbLineTrackInView)
                continue;

            std::cout << "Line ID " << pML->mnId
                    << " | P1_proj: (" << pML->mLsTrackProjX << ", " << pML->mLsTrackProjY << ")"
                    << " | P2_proj: (" << pML->mLeTrackProjX << ", " << pML->mLeTrackProjY << ")"
                    << " | Depth: " << pML->mLineTrackDepth
                    << " | Cos(view): " << pML->mLineTrackViewCos << std::endl;
        }

        // ========== [5] 显示窗口 ==========
        cv::imshow(winName, imDebug);
        cv::waitKey(0);
    }

    void LSDmatcher::DebugLineProjectedKeyFrame(KeyFrame* pKF, std::vector<MapLine*> &vpMapLines, const std::string &winName)
    {
        if(!pKF || vpMapLines.empty()) return;
        // 复制图像，用于绘制
        cv::Mat img;
        if(pKF->imgLeftRGB.channels() == 3)
            img = pKF->imgLeftRGB.clone();
        else
            cv::cvtColor(pKF->imgLeftRGB, img, cv::COLOR_GRAY2BGR);
        // 获取相机内参
        const float fx = pKF->fx;
        const float fy = pKF->fy;
        const float cx = pKF->cx;
        const float cy = pKF->cy;
        // 位姿
        Sophus::SE3f Tcw = pKF->GetPose();
        Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
        Eigen::Vector3f tcw = Tcw.translation();
        for(MapLine* pML : vpMapLines)
        {
            if(!pML || pML->isBad()) continue;
            Eigen::Vector3f SP, EP;
            std::tie(SP, EP) = pML->GetLineWorldPos();
            // 投影到相机坐标系
            Eigen::Vector3f SPc = Rcw * SP + tcw;
            Eigen::Vector3f EPc = Rcw * EP + tcw;
            // === 改进裁剪 Z <= 0 ===
            if(SPc(2) <= 0 && EPc(2) <= 0)
                continue; // 线段完全在相机后方，舍弃
            else if(SPc(2) <= 0)
            {
                float alpha = EPc(2) / (EPc(2)-SPc(2));
                SPc = SPc + alpha*(EPc-SPc);
            }
            else if(EPc(2) <= 0)
            {
                float alpha = SPc(2) / (SPc(2)-EPc(2));
                EPc = EPc + alpha*(SPc-EPc);
            }
            // 投影到像素平面
            cv::Point2f pt1(fx*SPc(0)/SPc(2) + cx, fy*SPc(1)/SPc(2) + cy);
            cv::Point2f pt2(fx*EPc(0)/EPc(2) + cx, fy*EPc(1)/EPc(2) + cy);
            // 可选：裁剪到图像边界
            pt1.x = std::max(0.f, std::min(pt1.x, float(pKF->mnMaxX)));
            pt1.y = std::max(0.f, std::min(pt1.y, float(pKF->mnMaxY)));
            pt2.x = std::max(0.f, std::min(pt2.x, float(pKF->mnMaxX)));
            pt2.y = std::max(0.f, std::min(pt2.y, float(pKF->mnMaxY)));
            // 画线
            cv::Scalar color(0, 0, 255); // 红色线
            cv::line(img, pt1, pt2, color, 2, cv::LINE_AA);
            // 可选：绘制端点
            cv::circle(img, pt1, 3, cv::Scalar(0,255,0), -1); // 绿色端点
            cv::circle(img, pt2, 3, cv::Scalar(255,0,0), -1); // 蓝色端点
        }
        // 显示
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::imshow(winName, img);
        cv::waitKey(1);
    }

    void LSDmatcher::DebugLineMatchesTwoFrames(KeyFrame* pKF1,KeyFrame* pKF2,const std::vector<std::pair<MapLine*, MapLine*>> &vpMatchedLines,const std::string &winName)
    {
        if(!pKF1 || !pKF2 || vpMatchedLines.empty()) return;
        // 克隆两帧图像
        cv::Mat img1, img2;
        if(pKF1->imgLeftRGB.channels() == 3)
            img1 = pKF1->imgLeftRGB.clone();
        else
            cv::cvtColor(pKF1->imgLeftRGB, img1, cv::COLOR_GRAY2BGR);
        if(pKF2->imgLeftRGB.channels() == 3)
            img2 = pKF2->imgLeftRGB.clone();
        else
            cv::cvtColor(pKF2->imgLeftRGB, img2, cv::COLOR_GRAY2BGR);
        // 相机内参
        const float fx1 = pKF1->fx, fy1 = pKF1->fy, cx1 = pKF1->cx, cy1 = pKF1->cy;
        const float fx2 = pKF2->fx, fy2 = pKF2->fy, cx2 = pKF2->cx, cy2 = pKF2->cy;
        // 位姿
        Sophus::SE3f Tcw1 = pKF1->GetPose();
        Sophus::SE3f Tcw2 = pKF2->GetPose();
        Eigen::Matrix3f Rcw1 = Tcw1.rotationMatrix();
        Eigen::Vector3f tcw1 = Tcw1.translation();
        Eigen::Matrix3f Rcw2 = Tcw2.rotationMatrix();
        Eigen::Vector3f tcw2 = Tcw2.translation();
        for(const auto &match : vpMatchedLines)
        {
            MapLine* pML1 = match.first;
            MapLine* pML2 = match.second;
            if(!pML1 || !pML2 || pML1->isBad() || pML2->isBad())
                continue;
            Eigen::Vector3f SP1, EP1, SP2, EP2;
            std::tie(SP1, EP1) = pML1->GetLineWorldPos();
            std::tie(SP2, EP2) = pML2->GetLineWorldPos();
            // 投影到相机坐标系
            Eigen::Vector3f SPc1 = Rcw1 * SP1 + tcw1;
            Eigen::Vector3f EPc1 = Rcw1 * EP1 + tcw1;
            Eigen::Vector3f SPc2 = Rcw2 * SP2 + tcw2;
            Eigen::Vector3f EPc2 = Rcw2 * EP2 + tcw2;
            // 简单裁剪 Z <= 0
            if(SPc1(2) <= 0 || EPc1(2) <= 0 || SPc2(2) <= 0 || EPc2(2) <= 0)
                continue;
            // 投影到像素坐标
            cv::Point2f pt1_1(fx1*SPc1(0)/SPc1(2) + cx1, fy1*SPc1(1)/SPc1(2) + cy1);
            cv::Point2f pt1_2(fx1*EPc1(0)/EPc1(2) + cx1, fy1*EPc1(1)/EPc1(2) + cy1);
            cv::Point2f pt2_1(fx2*SPc2(0)/SPc2(2) + cx2, fy2*SPc2(1)/SPc2(2) + cy2);
            cv::Point2f pt2_2(fx2*EPc2(0)/EPc2(2) + cx2, fy2*EPc2(1)/EPc2(2) + cy2);
            // 边界裁剪 C++11
            pt1_1.x = std::max(0.f, std::min(pt1_1.x, float(pKF1->mnMaxX)));
            pt1_1.y = std::max(0.f, std::min(pt1_1.y, float(pKF1->mnMaxY)));
            pt1_2.x = std::max(0.f, std::min(pt1_2.x, float(pKF1->mnMaxX)));
            pt1_2.y = std::max(0.f, std::min(pt1_2.y, float(pKF1->mnMaxY)));
            pt2_1.x = std::max(0.f, std::min(pt2_1.x, float(pKF2->mnMaxX)));
            pt2_1.y = std::max(0.f, std::min(pt2_1.y, float(pKF2->mnMaxY)));
            pt2_2.x = std::max(0.f, std::min(pt2_2.x, float(pKF2->mnMaxX)));
            pt2_2.y = std::max(0.f, std::min(pt2_2.y, float(pKF2->mnMaxY)));
            // 绘制 KF1 的线段
            cv::line(img1, pt1_1, pt1_2, cv::Scalar(0,0,255), 2, cv::LINE_AA);
            cv::circle(img1, pt1_1, 3, cv::Scalar(0,255,0), -1);
            cv::circle(img1, pt1_2, 3, cv::Scalar(255,0,0), -1);
            // 绘制 KF2 的线段
            cv::line(img2, pt2_1, pt2_2, cv::Scalar(0,255,255), 2, cv::LINE_AA);
            cv::circle(img2, pt2_1, 3, cv::Scalar(0,128,255), -1);
            cv::circle(img2, pt2_2, 3, cv::Scalar(255,128,0), -1);
            // 可选：画匹配连线（在合并显示时）
            cv::line(img1, pt1_1, pt2_1, cv::Scalar(255,255,0), 1, cv::LINE_AA);
            cv::line(img1, pt1_2, pt2_2, cv::Scalar(255,255,0), 1, cv::LINE_AA);
        }
        // 将两帧图像水平拼接显示
        cv::Mat imgConcat;
        cv::hconcat(img1, img2, imgConcat);
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::imshow(winName, imgConcat);
        cv::waitKey(1);
    }

    int LSDmatcher::SearchByDescriptor(KeyFrame* pKF, Frame &currentF, vector<MapLine*> &vpMapLineMatches)
    {
        if(currentF.NL==0 || pKF->NL==0)
            return 0;

        const std::vector<MapLine*> vpMapLinesKF = pKF->GetMapLineMatches();

        vpMapLineMatches = std::vector<MapLine*>(currentF.NL,static_cast<MapLine*>(NULL));

        int nmatches = 0;
        cv::BFMatcher* bfm = new cv::BFMatcher(cv::NORM_HAMMING, false);
        cv::Mat ldesc1, ldesc2;
        std::vector<std::vector<cv::DMatch>> lmatches;
        ldesc1 = pKF->mLineDescriptors;
        ldesc2 = currentF.mLineDescriptors;
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        std::cout<<"Initial LSDMATCH:lmatches"<<lmatches.size()<<endl;

        double nn_dist_th, nn12_dist_th;
        const float minRatio=1.0f/1.5f;
        currentF.lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th*0.5;
        sort(lmatches.begin(), lmatches.end(), LSDmatcher::sortByQueryIdx);
        for(int i=0; i<lmatches.size(); i++)
        {
            //lmatches里装的 是匹配的编号，
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;
            double dist_12 = lmatches[i][0].distance/lmatches[i][1].distance;
//            if(dist_12>nn12_dist_th)
            if(dist_12<minRatio)
            {
                MapLine* mapLine = vpMapLinesKF[qdx];
                if(mapLine)
                {
                    vpMapLineMatches[tdx]=mapLine;
                    nmatches++;
                }

            }
        }
        return nmatches;
    }

    int LSDmatcher::SearchByDescriptor(KeyFrame* pKF, KeyFrame *pKF2, vector<MapLine*> &vpMapLineMatches)
    {
        const std::vector<MapLine*> vpMapLinesKF = pKF->GetMapLineMatches();
        const std::vector<MapLine*> vpMapLinesKF2 = pKF2->GetMapLineMatches();

        vpMapLineMatches = std::vector<MapLine*>(vpMapLinesKF.size(),static_cast<MapLine*>(NULL));
        int nmatches = 0;
        cv::BFMatcher* bfm = new cv::BFMatcher(cv::NORM_HAMMING, false);
        cv::Mat ldesc1, ldesc2;
        vector<vector<DMatch>> lmatches;
        ldesc1 = pKF->mLineDescriptors;
        ldesc2 = pKF2->mLineDescriptors;
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        pKF2->lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th*0.5;
        sort(lmatches.begin(), lmatches.end(), LSDmatcher::sortByQueryIdx);
        for(int i=0; i<lmatches.size(); i++)
        {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;
            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if(dist_12>nn12_dist_th)
            {
                MapLine* mapLine = vpMapLinesKF2[tdx];
                if(mapLine) {
                    vpMapLineMatches[qdx] = mapLine;
                    nmatches++;
                }
            }
        }
        return nmatches;
    }

     int LSDmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                           std::vector<std::pair<int, int>> &vMatchedPairs)
    {
        vMatchedPairs.clear();
        int nmatches = 0;
        cv::BFMatcher* bfm = new cv::BFMatcher(cv::NORM_HAMMING, false);
        cv::Mat ldesc1, ldesc2;
        std::vector<std::vector<cv::DMatch>> lmatches;
        ldesc1 = pKF1->mLineDescriptors;
        ldesc2 = pKF2->mLineDescriptors;
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        pKF1->lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th*0.1;
        sort(lmatches.begin(), lmatches.end(), LSDmatcher::sortByQueryIdx);
        for(int i=0; i<lmatches.size(); i++)
        {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;
            if (pKF1->GetMapLine(qdx) || pKF2->GetMapLine(tdx)) {
                continue;
            }
            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if(dist_12>nn12_dist_th)
            {
                vMatchedPairs.push_back(make_pair(qdx, tdx));
                nmatches++;
            }
        }
        return nmatches;
    }

    void LSDmatcher::SearchForTriangulationLine(
        KeyFrame *pKF1, KeyFrame *pKF2,
        vector<pair<int,int>> &vMatchedIdx)
    {
        vMatchedIdx.clear();
        const auto &vLines1 = pKF1->mvKeyLines;
        const auto &vLines2 = pKF2->mvKeyLines;
        const int N1 = vLines1.size();
        const int N2 = vLines2.size();
        if (N1 == 0 || N2 == 0)
            return;
        // --- 预计算 KeyFrame2 的方向 ---
        vector<Eigen::Vector2f> vDirs2(N2);
        for (int i = 0; i < N2; i++)
        {
            Eigen::Vector2f d(
                vLines2[i].endPointX - vLines2[i].startPointX,
                vLines2[i].endPointY - vLines2[i].startPointY
            );
            vDirs2[i] = d.normalized();
        }
        const float cosAngleTh = cosf(20.f * M_PI / 180.f);
        for (int i1 = 0; i1 < N1; i1++)
        {
            const auto &l1 = vLines1[i1];
            Eigen::Vector2f d1(
                l1.endPointX - l1.startPointX,
                l1.endPointY - l1.startPointY);
            d1.normalize();
            Eigen::Vector2f s1(l1.startPointX, l1.startPointY);
            Eigen::Vector2f e1(l1.endPointX,   l1.endPointY);
            float bestDist = std::numeric_limits<float>::max();
            int bestIdx = -1;
            for (int i2 = 0; i2 < N2; i2++)
            {
            const auto &l2 = vLines2[i2];
            // (1) 方向（允许反向）
            float cosDir = fabs(d1.dot(vDirs2[i2]));
            if (cosDir < cosAngleTh)
                continue;
            // (2) 长度比例
            float len1 = l1.lineLength;
            float len2 = l2.lineLength;
            float lenDiff = fabs(len1 - len2) / std::max(len1, len2);
            if (lenDiff > 0.5f)
                continue;
            // (3) 两种端点排列
            Eigen::Vector2f s2(l2.startPointX, l2.startPointY);
            Eigen::Vector2f e2(l2.endPointX,   l2.endPointY);
            float dA = (s1 - s2).squaredNorm() + (e1 - e2).squaredNorm();
            float dB = (s1 - e2).squaredNorm() + (e1 - s2).squaredNorm();
            float dsum = std::min(dA, dB);
            if (dsum < bestDist)
            {
                bestDist = dsum;
                bestIdx = i2;
            }
        }
        if (bestIdx >= 0)
            vMatchedIdx.emplace_back(i1, bestIdx);
        }
    }

    int LSDmatcher::SearchForTriangulationFused(
        KeyFrame *pKF1, KeyFrame *pKF2,
        std::vector<std::pair<int, int>> &vMatchedPairs)
    {
        vMatchedPairs.clear();
        int nmatches = 0;
        const auto &vLines1 = pKF1->mvKeyLines;
        const auto &vLines2 = pKF2->mvKeyLines;
        const int N1 = vLines1.size();
        const int N2 = vLines2.size();
        if (N1 == 0 || N2 == 0)
            return 0;
        // === 1. 预计算 KeyFrame2 的方向向量 ===
        std::vector<Eigen::Vector2f> vDirs2(N2);
        for (int i = 0; i < N2; i++)
        {
            Eigen::Vector2f d(
                vLines2[i].endPointX - vLines2[i].startPointX,
                vLines2[i].endPointY - vLines2[i].startPointY);
            vDirs2[i] = d.normalized();
        }
        const float cosAngleTh = cosf(20.f * M_PI / 180.f);  // 最大方向差 20°
        // === 2. 几何候选匹配 ===
        std::vector<std::pair<int,int>> geomCandidates;
        for (int i1 = 0; i1 < N1; i1++)
        {
            const auto &l1 = vLines1[i1];
            Eigen::Vector2f d1(
                l1.endPointX - l1.startPointX,
                l1.endPointY - l1.startPointY);
            d1.normalize();
            Eigen::Vector2f s1(l1.startPointX, l1.startPointY);
            Eigen::Vector2f e1(l1.endPointX,   l1.endPointY);
            float bestDist = std::numeric_limits<float>::max();
            int bestIdx = -1;
            for (int i2 = 0; i2 < N2; i2++)
            {
                const auto &l2 = vLines2[i2];
                // 方向允许反向
                float cosDir = fabs(d1.dot(vDirs2[i2]));
                if (cosDir < cosAngleTh)
                    continue;
                // 长度比例
                float len1 = l1.lineLength;
                float len2 = l2.lineLength;
                float lenDiff = fabs(len1 - len2) / std::max(len1, len2);
                if (lenDiff > 0.5f)
                    continue;
                // 两种端点排列
                Eigen::Vector2f s2(l2.startPointX, l2.startPointY);
                Eigen::Vector2f e2(l2.endPointX,   l2.endPointY);
                float dA = (s1 - s2).squaredNorm() + (e1 - e2).squaredNorm();
                float dB = (s1 - e2).squaredNorm() + (e1 - s2).squaredNorm();
                float dsum = std::min(dA, dB);
                if (dsum < bestDist)
                {
                    bestDist = dsum;
                    bestIdx = i2;
                }
            }
            if (bestIdx >= 0)
                geomCandidates.emplace_back(i1, bestIdx);
        }
        if (geomCandidates.empty())
            return 0;
        // === 3. 描述子验证 ===
        cv::BFMatcher bfm(cv::NORM_HAMMING, false);
        std::vector<std::vector<cv::DMatch>> knnMatches;
        cv::Mat ldesc1 = pKF1->mLineDescriptors;
        cv::Mat ldesc2 = pKF2->mLineDescriptors;
        bfm.knnMatch(ldesc1, ldesc2, knnMatches, 2);
        // 计算 MAD 阈值
        double nn_dist_th, nn12_dist_th;
        pKF1->lineDescriptorMAD(knnMatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th *= 0.1; // 调整比例
        // 候选集合筛选
        for (auto &pair : geomCandidates)
        {
            int qdx = pair.first;
            int tdx = pair.second;
            if (pKF1->GetMapLine(qdx) || pKF2->GetMapLine(tdx))
                continue;
            // 找在 knnMatches 中对应的 trainIdx
            for (auto &m : knnMatches[qdx])
            {
                if (m.trainIdx == tdx)
                {
                    if (knnMatches[qdx].size() > 1)
                    {
                        double dist12 = knnMatches[qdx][1].distance - knnMatches[qdx][0].distance;
                        if (dist12 > nn12_dist_th)
                        {
                            vMatchedPairs.emplace_back(qdx, tdx);
                            nmatches++;
                        }
                    }
                    else
                    {
                        // 只有一个匹配时也保留
                        vMatchedPairs.emplace_back(qdx, tdx);
                        nmatches++;
                    }
                    break;
                }
            }
        }
        return nmatches;
    }


    // int LSDmatcher::Fuse(KeyFrame *pKF, const std::vector<MapLine *> &vpMapLines, const float th)
    // {
    //     Eigen::Matrix3f Rcw_eigen = pKF->GetRotation();
    //     Eigen::Vector3f tcw_eigen = pKF->GetTranslation();
    //     Eigen::Vector3f Ow_eigen = pKF->GetCameraCenter();
    //     cv::Mat Rcw = Converter::toCvMat(Rcw_eigen);
    //     cv::Mat tcw = (cv::Mat_<float>(3,1) << tcw_eigen(0), tcw_eigen(1), tcw_eigen(2));
    //     const float &fx = pKF->fx;
    //     const float &fy = pKF->fy;
    //     const float &cx = pKF->cx;
    //     const float &cy = pKF->cy;
    //     int nFused = 0;
    //     const int nLines = vpMapLines.size();
    //     for(int iML = 0; iML < nLines; iML++)
    //     {
    //         MapLine* pML = vpMapLines[iML];
    //         if(!pML || pML->isBad())
    //             continue;
    //         // 获取世界坐标线段端点
    //         std::pair<Eigen::Vector3f, Eigen::Vector3f> lineWorldPos = pML->GetLineWorldPos();
    //         Eigen::Vector3f SP = lineWorldPos.first;
    //         Eigen::Vector3f EP = lineWorldPos.second;
    //         // 投影到相机坐标系
    //         Eigen::Vector3f SPc = Rcw_eigen * SP + tcw_eigen;
    //         Eigen::Vector3f EPc = Rcw_eigen * EP + tcw_eigen;
    //         // 端点深度检查
    //         if(SPc(2) <= 0 || EPc(2) <= 0)
    //             continue;
    //         // 计算像素坐标
    //         float u1 = fx * SPc(0)/SPc(2) + cx;
    //         float v1 = fy * SPc(1)/SPc(2) + cy;
    //         float u2 = fx * EPc(0)/EPc(2) + cx;
    //         float v2 = fy * EPc(1)/EPc(2) + cy;
    //         if(u1 < pKF->mnMinX || u1 > pKF->mnMaxX || v1 < pKF->mnMinY || v1 > pKF->mnMaxY)
    //             continue;
    //         if(u2 < pKF->mnMinX || u2 > pKF->mnMaxX || v2 < pKF->mnMinY || v2 > pKF->mnMaxY)
    //             continue;
    //         // 距离限制
    //         const float maxDistance = pML->GetMaxDistanceInvariance();
    //         const float minDistance = pML->GetMinDistanceInvariance();
    //         Eigen::Vector3f OM = 0.5f*(SP+EP) - Ow_eigen;
    //         float dist = OM.norm();
    //         if(dist < minDistance || dist > maxDistance)
    //             continue;
    //         // === 可见性判断：相机朝向与线段中点向量夹角 ===
    //         Eigen::Vector3f camDir = (Eigen::Vector3f(0,0,1).transpose() * Rcw_eigen).transpose(); // 相机前向
    //         Eigen::Vector3f OMdir = OM.normalized();
    //         float cosAngle = camDir.dot(OMdir);
    //         if(cosAngle < 0.3f) // 阈值可调
    //             continue;
    //         // 尺度预测
    //         int nPredictedLevel = pML->PredictScale(dist, pKF);
    //         float radius = th * pKF->mvScaleFactors[nPredictedLevel];
    //         // 候选 KeyLine
    //         std::vector<std::size_t> vIndices = pKF->GetLinesInArea(u1,v1, u2,v2, radius);
    //         if(vIndices.empty())
    //             continue;
    //         // 描述子匹配
    //         cv::Mat dML = pML->GetLineDescriptor();
    //         int bestDist = INT_MAX;
    //         int bestIdx = -1;
    //         for(size_t idx : vIndices)
    //         {
    //             int klLevel = pKF->mvKeyLines[idx].octave;
    //             if(klLevel < nPredictedLevel-1 || klLevel > nPredictedLevel)
    //                 continue;
    //             const cv::Mat &dKF = pKF->mLineDescriptors.row(idx);
    //             int distDesc = DescriptorDistance(dML,dKF);
    //             if(distDesc < bestDist)
    //             {
    //                 bestDist = distDesc;
    //                 bestIdx = idx;
    //             }
    //         }
    //         if(bestDist <= TH_LOW)
    //         {
    //             MapLine* pMLinKF = pKF->GetMapLine(bestIdx);
    //             if(pMLinKF)
    //             {
    //                 if(!pMLinKF->isBad())
    //                 {
    //                     if(pMLinKF->Observations() > pML->Observations())
    //                         pML->Replace(pMLinKF);
    //                     else
    //                         pMLinKF->Replace(pML);
    //                 }
    //             }
    //             else
    //             {
    //                 pML->AddLineObservation(pKF, bestIdx);
    //                 pKF->AddMapLine(pML, bestIdx);
    //             }
    //             nFused++;
    //         }
    //     }
    //     return nFused;
    // }

    int LSDmatcher::Fuse(KeyFrame *pKF, const std::vector<MapLine *> &vpMapLines, const float th)
    {
        Eigen::Matrix3f Rcw = pKF->GetRotation();
        Eigen::Vector3f tcw = pKF->GetTranslation();
        Eigen::Vector3f Ow = pKF->GetCameraCenter();
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        int nFused = 0;
        for (MapLine* pML : vpMapLines)
        {
            if (!pML || pML->isBad())
                continue;
            auto [SP, EP] = pML->GetLineWorldPos();
            // 投影到相机坐标系
            Eigen::Vector3f SPc = Rcw * SP + tcw;
            Eigen::Vector3f EPc = Rcw * EP + tcw;
            // === 在相机坐标系下裁剪 Z <= 0 ===
            Eigen::Vector3f SPc_clip = SPc;
            Eigen::Vector3f EPc_clip = EPc;
            if (SPc_clip(2) <= 0 && EPc_clip(2) > 0)
            {
                float alpha = EPc_clip(2) / (EPc_clip(2)-SPc_clip(2));
                SPc_clip = SPc_clip + alpha*(EPc_clip-SPc_clip);
            }
            else if (EPc_clip(2) <= 0 && SPc_clip(2) > 0)
            {
                float alpha = SPc_clip(2) / (SPc_clip(2)-EPc_clip(2));
                EPc_clip = EPc_clip + alpha*(SPc_clip-EPc_clip);
            }
            else if (SPc_clip(2) <= 0 && EPc_clip(2) <= 0)
                continue; // 整条线在相机后方
            // 投影到像素坐标
            float u1 = fx * SPc_clip(0)/SPc_clip(2) + cx;
            float v1 = fy * SPc_clip(1)/SPc_clip(2) + cy;
            float u2 = fx * EPc_clip(0)/EPc_clip(2) + cx;
            float v2 = fy * EPc_clip(1)/EPc_clip(2) + cy;
            // 裁剪到图像边界
            // u1 = std::clamp(u1, (float)pKF->mnMinX, (float)pKF->mnMaxX);
            // v1 = std::clamp(v1, (float)pKF->mnMinY, (float)pKF->mnMaxY);
            // u2 = std::clamp(u2, (float)pKF->mnMinX, (float)pKF->mnMaxX);
            // v2 = std::clamp(v2, (float)pKF->mnMinY, (float)pKF->mnMaxY);
            // 替换后
            if (u1 < pKF->mnMinX) u1 = (float)pKF->mnMinX;
            if (u1 > pKF->mnMaxX) u1 = (float)pKF->mnMaxX;
            if (v1 < pKF->mnMinY) v1 = (float)pKF->mnMinY;
            if (v1 > pKF->mnMaxY) v1 = (float)pKF->mnMaxY;

            if (u2 < pKF->mnMinX) u2 = (float)pKF->mnMinX;
            if (u2 > pKF->mnMaxX) u2 = (float)pKF->mnMaxX;
            if (v2 < pKF->mnMinY) v2 = (float)pKF->mnMinY;
            if (v2 > pKF->mnMaxY) v2 = (float)pKF->mnMaxY;

            // // 距离限制
            float dist = (0.5f*(SP+EP) - Ow).norm();
            // if(dist < pML->GetMinDistanceInvariance() || dist > pML->GetMaxDistanceInvariance())
            //     continue;
            // 尺度预测和半径
            int nPredictedLevel = pML->PredictScale(dist, pKF);
            float radius = th * pKF->mvScaleFactors[nPredictedLevel];
            // 获取候选KeyLine
            std::vector<std::size_t> vIndices = pKF->GetLinesInArea(u1,v1,u2,v2,radius);
            if(vIndices.empty())
                continue;
            // 描述子匹配
            cv::Mat dML = pML->GetLineDescriptor();
            int bestDist = INT_MAX;
            int bestIdx = -1;
            for (size_t idx : vIndices)
            {
                int klLevel = pKF->mvKeyLines[idx].octave;
                if(klLevel < nPredictedLevel-1 || klLevel > nPredictedLevel)
                    continue;
                const cv::Mat &dKF = pKF->mLineDescriptors.row(idx);
                int distDesc = DescriptorDistance(dML, dKF);
                if(distDesc < bestDist)
                {
                    bestDist = distDesc;
                    bestIdx = idx;
                }
            }
            if(bestDist <= TH_LOW)
            {
                MapLine* pMLinKF = pKF->GetMapLine(bestIdx);
                if(pMLinKF)
                {
                    if(!pMLinKF->isBad())
                    {
                        if(pMLinKF->Observations() > pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                }
                else
                {
                    pML->AddLineObservation(pKF, bestIdx);
                    pKF->AddMapLine(pML, bestIdx);
                }
                nFused++;
            }
        }
        return nFused;
    }

    int LSDmatcher::FuseOld(KeyFrame* pKF, const std::vector<MapLine*>& vpMapLines, const float th)
    {
        Eigen::Matrix3f Rcw = pKF->GetRotation();
        Eigen::Vector3f tcw = pKF->GetTranslation();
        Eigen::Vector3f Ow = pKF->GetCameraCenter();
        const float& fx = pKF->fx;
        const float& fy = pKF->fy;
        const float& cx = pKF->cx;
        const float& cy = pKF->cy;
        int nFused = 0;
        std::unordered_set<MapLine*> fusedSet;
        for(MapLine* pML : vpMapLines)
        {
            if(!pML || pML->isBad() || fusedSet.count(pML))
                continue;
            fusedSet.insert(pML);
            // Step 1: 获取数据，无锁
            auto [SP, EP] = pML->GetLineWorldPos();
            float dist = (0.5f*(SP+EP)-Ow).norm();
            int nPredictedLevel = pML->PredictScale(dist, pKF);
            float radius = th * pKF->mvScaleFactors[nPredictedLevel];
            // 投影到相机坐标
            Eigen::Vector3f SPc = Rcw*SP + tcw;
            Eigen::Vector3f EPc = Rcw*EP + tcw;
            // 裁剪相机后方
            if(SPc(2)<=0 && EPc(2)<=0) continue;
            if(SPc(2)<=0) SPc += (EPc-SPc)*(EPc(2)/(EPc(2)-SPc(2)));
            else if(EPc(2)<=0) EPc += (SPc-EPc)*(SPc(2)/(SPc(2)-EPc(2)));
            // 投影到像素
            float u1 = fx*SPc(0)/SPc(2)+cx, v1 = fy*SPc(1)/SPc(2)+cy;
            float u2 = fx*EPc(0)/EPc(2)+cx, v2 = fy*EPc(1)/EPc(2)+cy;
            u1 = std::max((float)pKF->mnMinX, std::min(u1, (float)pKF->mnMaxX));
            v1 = std::max((float)pKF->mnMinY, std::min(v1, (float)pKF->mnMaxY));
            u2 = std::max((float)pKF->mnMinX, std::min(u2, (float)pKF->mnMaxX));
            v2 = std::max((float)pKF->mnMinY, std::min(v2, (float)pKF->mnMaxY));
            std::vector<size_t> vIndices = pKF->GetLinesInArea(u1,v1,u2,v2,radius);
            if(vIndices.empty()) continue;
            cv::Mat dML = pML->GetLineDescriptor();
            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(size_t idx : vIndices)
            {
                int klLevel = pKF->mvKeyLines[idx].octave;
                if(klLevel<nPredictedLevel-1 || klLevel>nPredictedLevel) continue;
                const cv::Mat& dKF = pKF->mLineDescriptors.row(idx);
                int distDesc = DescriptorDistance(dML,dKF);
                if(distDesc<bestDist){ bestDist=distDesc; bestIdx=idx; }
            }
            if(bestDist > TH_LOW) continue;
            // Step 2: 修改 MapLine 或 KeyFrame 状态，加锁最小粒度
            MapLine* currentML = pML;
            // 获取 KeyFrame 中对应 MapLine
            MapLine* pMLinKF = nullptr;
            {
                //std::unique_lock<std::mutex> lockKF(pKF->mMutexFeatures);
                pMLinKF = pKF->GetMapLine(bestIdx);
            }
            if(pMLinKF && !pMLinKF->isBad())
            {
                // 锁住两条 MapLine，顺序固定: 小ID先锁
                MapLine *first = (currentML->mnId < pMLinKF->mnId) ? currentML : pMLinKF;
                MapLine *second = (currentML->mnId < pMLinKF->mnId) ? pMLinKF : currentML;
                //std::unique_lock<std::mutex> lock1(first->mMutexPos);
                //std::unique_lock<std::mutex> lock2(second->mMutexPos);
                if(pMLinKF->Observations() > currentML->Observations())
                    currentML->Replace(pMLinKF);
                else
                    pMLinKF->Replace(currentML);
                currentML = currentML->GetReplaced();
            }
            else
            {
                //std::unique_lock<std::mutex> lockML(currentML->mMutexFeatures);
                currentML->AddLineObservation(pKF,bestIdx);
                {
                    //std::unique_lock<std::mutex> lockKF(pKF->mMutexFeatures);
                    pKF->AddMapLine(currentML,bestIdx);
                }
            }
            nFused++;
        }
        return nFused;
    }


    float LSDmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if(viewCos>0.998)
            return 5.0;
        else
            return 8.0;
    }

#if 0   //to do next...

    
    

    int LSDmatcher::SerachForInitialize(Frame &InitialFrame, Frame &CurrentFrame, vector<pair<int, int>> &LineMatches)
    {
        LineMatches.clear();
        int nmatches = 0;
        BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
        Mat ldesc1, ldesc2;
        vector<vector<DMatch>> lmatches;
        ldesc1 = InitialFrame.mLdesc;
        ldesc2 = CurrentFrame.mLdesc;
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        CurrentFrame.lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th*0.5;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
        for(int i=0; i<lmatches.size(); i++)
        {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;
            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if(dist_12>nn12_dist_th)
            {
                LineMatches.push_back(make_pair(qdx, tdx));
                nmatches++;
            }
        }
        return nmatches;
    }

    
   
    int LSDmatcher::SearchByProjection(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapLine *> &vpLines,
                                       std::vector<MapLine *> &vpMatched, int th) {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
        const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        cv::Mat Rcw = sRcw/scw;
        cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
        cv::Mat Ow = -Rcw.t()*tcw;

        // Set of MapLines already found in the KeyFrame
        set<MapLine*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapLine*>(NULL));

        int nmatches=0;

        // For each Candidate MapLine Project and Match
        for(int iML=0, iendML=vpLines.size(); iML<iendML; iML++)
        {
            MapLine* pML = vpLines[iML];

            // Discard Bad MapLines and already found
            if(!pML || pML->isBad() || spAlreadyFound.count(pML))
                continue;


            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const auto &SPcX = SPc.at<float>(0);
            const auto &SPcY = SPc.at<float>(1);
            const auto &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const auto &EPcX = EPc.at<float>(0);
            const auto &EPcY = EPc.at<float>(1);
            const auto &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if (u1 < pKF->mnMinX || u1 > pKF->mnMaxX)
                continue;
            if (v1 < pKF->mnMinY || v1 > pKF->mnMaxY)
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            if (u2 < pKF->mnMinX || u2 > pKF->mnMaxX)
                continue;
            if (v2 < pKF->mnMinY || v2 > pKF->mnMaxY)
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const cv::Mat OM = 0.5 * (SP + EP) - Ow;
            const float dist = cv::norm(OM);

            if (dist < minDistance || dist > maxDistance)
                continue;

            Vector3d Pn = pML->GetNormal();
            cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));

            if(OM.dot(pn)<0.5*dist)
                continue;

            const int nPredictedLevel = pML->PredictScale(dist, pKF->mfLogScaleFactor);

            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetLinesInArea(u1,v1, u2, v2, radius);

            if(vIndices.empty())
                continue;

            const cv::Mat dML = pML->GetDescriptor();

            int bestDist=256;
            int bestIdx =-1 ;

            for(unsigned long idx : vIndices)
            {
                if(vpMatched[idx])
                    continue;

                const int &klLevel = pKF->mvKeyLines[idx].octave;

                if(klLevel<nPredictedLevel-1 || klLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(dML,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW)
            {
                vpMatched[bestIdx]=pML;
                nmatches++;
            }
        }

        return nmatches;
    }

    int LSDmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapLine *> &vpMatches12, const float &s12,
                                 const cv::Mat &R12, const cv::Mat &t12, const float th) {
        const float &fx = pKF1->fx;
        const float &fy = pKF1->fy;
        const float &cx = pKF1->cx;
        const float &cy = pKF1->cy;

        // Camera 1 from world
        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();

        //Camera 2 from world
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        //Transformation between cameras
        cv::Mat sR12 = s12*R12;
        cv::Mat sR21 = (1.0/s12)*R12.t();
        cv::Mat t21 = -sR21*t12;

        const vector<MapLine*> vpMapLines1 = pKF1->GetMapLineMatches();
        const int N1 = vpMapLines1.size();

        const vector<MapLine*> vpMapLines2 = pKF2->GetMapLineMatches();
        const int N2 = vpMapLines2.size();

        vector<bool> vbAlreadyMatched1(N1,false);
        vector<bool> vbAlreadyMatched2(N2,false);

        for(int i=0; i<N1; i++)
        {
            MapLine* pML = vpMatches12[i];
            if(pML)
            {
                vbAlreadyMatched1[i]=true;
                int idx2 = pML->GetIndexInKeyFrame(pKF2);
                if(idx2>=0 && idx2<N2)
                    vbAlreadyMatched2[idx2]=true;
            }
        }

        vector<int> vnMatch1(N1,-1);
        vector<int> vnMatch2(N2,-1);

        // Transform from KF1 to KF2 and search
        for(int i1=0; i1<N1; i1++)
        {
            MapLine* pML = vpMapLines1[i1];

            if(!pML || pML->isBad() || vbAlreadyMatched1[i1])
                continue;

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc1 = R1w * SP + t1w;
            const cv::Mat SPc2 = sR21 * SPc1 + t21;
            const auto &SPcX = SPc2.at<float>(0);
            const auto &SPcY = SPc2.at<float>(1);
            const auto &SPcZ = SPc2.at<float>(2);

            const cv::Mat EPc1 = R1w * EP + t1w;
            const cv::Mat EPc2 = sR21 * EPc1 + t21;
            const auto &EPcX = EPc2.at<float>(0);
            const auto &EPcY = EPc2.at<float>(1);
            const auto &EPcZ = EPc2.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if(!pKF2->IsInImage(u1,v1))
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            if(!pKF2->IsInImage(u2,v2))
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const float dist3D = cv::norm(0.5 * (SPc2 + EPc2));

            if (dist3D < minDistance || dist3D > maxDistance) {
                continue;
            }

            // Compute predicted octave
            const int nPredictedLevel = pML->PredictScale(dist3D, pKF2->mfLogScaleFactor);

            // Search in a radius
            const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF2->GetLinesInArea(u1,v1,u2,v2,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dML = pML->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(unsigned long idx : vIndices)
            {
                const int &klLevel = pKF2->mvKeyLines[idx].octave;

                if(klLevel<nPredictedLevel-1 || klLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF2->mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(dML, dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_HIGH)
            {
                vnMatch1[i1]=bestIdx;
            }
        }

        // Transform from KF2 to KF1 and search
        for(int i2=0; i2<N2; i2++)
        {
            MapLine* pML = vpMapLines2[i2];

            if(!pML || pML->isBad() || vbAlreadyMatched2[i2])
                continue;

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc2 = R2w * SP + t2w;
            const cv::Mat SPc1 = sR12 * SPc2 + t12;
            const auto &SPcX = SPc1.at<float>(0);
            const auto &SPcY = SPc1.at<float>(1);
            const auto &SPcZ = SPc1.at<float>(2);

            const cv::Mat EPc2 = R2w * EP + t2w;
            const cv::Mat EPc1 = sR12 * EPc2 + t12;
            const auto &EPcX = EPc1.at<float>(0);
            const auto &EPcY = EPc1.at<float>(1);
            const auto &EPcZ = EPc1.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if(!pKF1->IsInImage(u1,v1))
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            if(!pKF1->IsInImage(u2,v2))
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const float dist3D = cv::norm(0.5 * (SPc1 + EPc1));

            if (dist3D < minDistance || dist3D > maxDistance)
                continue;

            // Compute predicted octave
            const int nPredictedLevel = pML->PredictScale(dist3D, pKF1->mfLogScaleFactor);

            // Search in a radius
            const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF1->GetLinesInArea(u1,v1,u2,v2,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dML = pML->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(unsigned long idx : vIndices)
            {
                const int &klLevel = pKF1->mvKeyLines[idx].octave;

                if(klLevel<nPredictedLevel-1 || klLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF1->mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(dML, dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_HIGH)
            {
                vnMatch2[i2]=bestIdx;
            }
        }

        // Check agreement
        int nFound = 0;

        for(int i1=0; i1<N1; i1++)
        {
            int idx2 = vnMatch1[i1];

            if(idx2>=0)
            {
                int idx1 = vnMatch2[idx2];
                if(idx1==i1)
                {
                    vpMatches12[i1] = vpMapLines2[idx2];
                    nFound++;
                }
            }
        }

        return nFound;
    }

    int LSDmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapLine *> &vpLines, float th,
                         vector<MapLine *> &vpReplaceLine) {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
        const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        cv::Mat Rcw = sRcw/scw;
        cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
        cv::Mat Ow = -Rcw.t()*tcw;

        // Set of MapPoints already found in the KeyFrame
        const set<MapLine*> spAlreadyFound = pKF->GetMapLines();

        int nFused=0;

        const int nLines = vpLines.size();

        // For each candidate MapPoint project and match
        for(int iML=0; iML<nLines; iML++)
        {
            MapLine* pML = vpLines[iML];

            // Discard Bad MapPoints and already found
            if(!pML || pML->isBad() || spAlreadyFound.count(pML))
                continue;

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const auto &SPcX = SPc.at<float>(0);
            const auto &SPcY = SPc.at<float>(1);
            const auto &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const auto &EPcX = EPc.at<float>(0);
            const auto &EPcY = EPc.at<float>(1);
            const auto &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if (u1 < pKF->mnMinX || u1 > pKF->mnMaxX)
                continue;
            if (v1 < pKF->mnMinY || v1 > pKF->mnMaxY)
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            if (u2 < pKF->mnMinX || u2 > pKF->mnMaxX)
                continue;
            if (v2 < pKF->mnMinY || v2 > pKF->mnMaxY)
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const cv::Mat OM = 0.5 * (SP + EP) - Ow;
            const float dist = cv::norm(OM);

            if (dist < minDistance || dist > maxDistance)
                continue;

            Vector3d Pn = pML->GetNormal();
            cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));

            if(OM.dot(pn)<0.5*dist)
                continue;

            const int nPredictedLevel = pML->PredictScale(dist, pKF->mfLogScaleFactor);

            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetLinesInArea(u1,v1, u2, v2, radius);

            if(vIndices.empty())
                continue;

            const cv::Mat dML = pML->GetDescriptor();

            int bestDist=INT_MAX;
            int bestIdx =-1 ;

            for(unsigned long idx : vIndices)
            {
                const int &klLevel = pKF->mvKeyLines[idx].octave;

                if(klLevel<nPredictedLevel-1 || klLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(dML,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW)
            {
                MapLine* pMLinKF = pKF->GetMapLine(bestIdx);
                if(pMLinKF)
                {
                    if(!pMLinKF->isBad())
                        vpReplaceLine[iML] = pMLinKF;
                }
                else
                {
                    pML->AddObservation(pKF,bestIdx);
                    pKF->AddMapLine(pML,bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

#endif

}
