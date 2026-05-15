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

    int LSDmatcher::SearchByProjectionNewKeyFrame(KeyFrame* pKF, Frame &currentF, std::vector<MapLine*> &vpMapLineMatches)
    {
        if(currentF.NL == 0 || pKF->NL == 0)
            return 0;
        int nmatches = 0;
        vpMapLineMatches.assign(currentF.NL, nullptr);

        // 获取当前帧相对于参考帧/地图的位姿 (用于后续 isInFrustum 等判断)
        // 假设 currentF 已经由 Motion Model 设置了初步位姿

        // 遍历参考 KeyFrame 中的所有 MapLine
        const std::vector<MapLine*> vpMapLinesKF = pKF->GetMapLineMatches();

        for (size_t iKF = 0; iKF < vpMapLinesKF.size(); iKF++)
        {
            MapLine* pML = vpMapLinesKF[iKF];
            if (!pML || pML->isBad()) continue;

            // 1. 【几何过滤】检查线段是否在当前帧的视锥体内 (Frustum Culling)
            // 这个函数会计算线段在当前帧的投影位置 mLsTrackProjX, mLsTrackProjY 等
            if (!currentF.isLineInFrustum(pML, 0.5f)) 
                continue;

            // 获取投影后的参数 (投影直线方程或端点)
            float projStartX = pML->mLsTrackProjX;
            float projStartY = pML->mLsTrackProjY;
            float projEndX = pML->mLeTrackProjX;
            float projEndY = pML->mLeTrackProjY;

            // 计算投影线段的角度 (用于法向约束)
            float angleProj = atan2(projEndY - projStartY, projEndX - projStartX);

            // 2. 【局部搜索】在投影位置附近的网格中寻找候选线段
            // 假设你已经像 ORB 那样把线段按坐标分配到了 grid 里
            float radius = 50.0f; // 搜索半径（像素）
            vector<size_t> vIndices = currentF.GetLinesInRegion(projStartX, projStartY, projEndX, projEndY, radius);

            if (vIndices.empty()) continue;

            const cv::Mat& dKF = pKF->mLineDescriptors.row(iKF);
            int bestDist = 256;
            int bestIdx = -1;

            for (const size_t iF : vIndices)
            {
                if (vpMapLineMatches[iF]) continue; // 已经被匹配过了

                const cv::line_descriptor::KeyLine& kl = currentF.mvKeyLinesUn[iF];

                // 3. 【方向约束 (核心)】 角度差必须小于 10 度
                float dx = kl.endPointX - kl.startPointX;
                float dy = kl.endPointY - kl.startPointY;
                float angleDetected = atan2(dy, dx);
            
                float angleDiff = fabs(angleProj - angleDetected);
                if (angleDiff > M_PI) angleDiff = 2 * M_PI - angleDiff;
                // 兼容线段方向反向的情况 (0度和180度等效)
                if (angleDiff > M_PI_2) angleDiff = fabs(M_PI - angleDiff);

                if (angleDiff > 10.0f * M_PI / 180.0f) 
                    continue; 

                // 4. 【描述子匹配】在通过几何校验的候选集中找最像的
                const cv::Mat& dF = currentF.mLineDescriptors.row(iF);
                int dist = DescriptorDistance(dKF, dF); // 计算汉明距离

                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = iF;
                }
            }

            // 5. 【阈值检查】
            if (bestDist < 64) // 经验阈值
            {
                vpMapLineMatches[bestIdx] = pML;
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

    //收集所有的初步匹配对，计算它们的像素位移，然后剔除偏离中位数的“离群值”
    int LSDmatcher::SearchByProjectionDisplacement(Frame &F, const std::vector<MapLine *> &vpMapLines, const float th)
    {
        int nmatches = 0;
        const bool bFactor = (th != 1.0f);

        // 🌟 新增：记录匹配对及其像素位移，用于一致性检查
        struct MatchCandidate {
            size_t frameLineIdx;
            MapLine* pMapLine;
            float displacementX;
            float displacementY;
        };
        std::vector<MatchCandidate> vCandidates;

        for (auto pML : vpMapLines)
        {
            if (!pML || pML->isBad() || !pML->mbLineTrackInView)
                continue;

            const int nPredictLevel = pML->mnLineTrackScaleLevel;
            float r = RadiusByViewingCos(pML->mLineTrackViewCos);
            if (bFactor) r *= th;

            const float u_mean_proj = 0.5f * (pML->mLsTrackProjX + pML->mLeTrackProjX);
            const float v_mean_proj = 0.5f * (pML->mLsTrackProjY + pML->mLeTrackProjY);

            const int minLevel = std::max(0, nPredictLevel - 1);
            const int maxLevel = std::min(nPredictLevel + 1, F.mnScaleLevels - 1);

            // 这里的半径适当加大以捕捉可能的位移
            std::vector<size_t> vIndices = F.GetLinesInAreaMean(u_mean_proj, v_mean_proj, 
                                        r * F.mvScaleFactors[nPredictLevel] * 2, minLevel, maxLevel);

            if (vIndices.empty()) continue;

            const cv::Mat MLdescriptor = pML->GetLineDescriptor();
            int bestDist = 256;
            int bestIdx = -1;

            for (size_t idx : vIndices) {
                if (F.mvpMapLines[idx]) continue;
                const cv::Mat &d = F.mLineDescriptors.row(idx);
                const int dist = DescriptorDistance(MLdescriptor, d);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // 初步筛选
            if (bestDist <= TH_HIGH) {
                const auto &kl = F.mvKeyLinesUn[bestIdx];
                float u_mean_det = 0.5f * (kl.startPointX + kl.endPointX);
                float v_mean_det = 0.5f * (kl.startPointY + kl.endPointY);

                // 🌟 记录该匹配的“预测位置”与“观测位置”的像素偏移
                vCandidates.push_back({
                    (size_t)bestIdx, 
                    pML, 
                    u_mean_det - u_mean_proj, 
                    v_mean_det - v_mean_proj
                });
            }
        }

        if (vCandidates.empty()) return 0;

        // 🌟 核心策略：全局偏移一致性检查 (Global Displacement Consistency)
        // 百叶窗错行匹配的偏移量通常会比正常匹配大几十个像素
    
        // 1. 计算位移的中位数（中位数对错行匹配非常鲁棒）
        std::vector<float> vDX, vDY;
        for(auto &c : vCandidates) {
            vDX.push_back(c.displacementX);
            vDY.push_back(c.displacementY);
        }
        std::sort(vDX.begin(), vDX.end());
        std::sort(vDY.begin(), vDY.end());
        float medianDX = vDX[vDX.size()/2];
        float medianDY = vDY[vDY.size()/2];

        // 2. 剔除偏离中位数过大的匹配对
        const float thConsistency = 20.0f; // 阈值设为 20 像素，百叶窗错行通常 > 40px
        for (auto &c : vCandidates) {
            float dev = std::sqrt(std::pow(c.displacementX - medianDX, 2) + 
                              std::pow(c.displacementY - medianDY, 2));
        
            if (dev < thConsistency) {
                // 只有偏移方向符合大部队的才真正记录为匹配
                F.mvpMapLines[c.frameLineIdx] = c.pMapLine;
                c.pMapLine->IncreaseFound();
                nmatches++;
            }
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

    // 假设 kl1 是地图线段在当前帧的投影，kl2 是当前帧检测到的线段
    bool LSDmatcher::CheckDirectionConsistency(const cv::line_descriptor::KeyLine& kl1, 
                                           const cv::line_descriptor::KeyLine& kl2, 
                                           float angle_threshold_deg)
    {
        // 1. 计算线段 1 的方向向量
        float dx1 = kl1.endPointX - kl1.startPointX;
        float dy1 = kl1.endPointY - kl1.startPointY;
        float len1 = sqrt(dx1*dx1 + dy1*dy1 + 1e-6f);
    
        // 2. 计算线段 2 的方向向量
        float dx2 = kl2.endPointX - kl2.startPointX;
        float dy2 = kl2.endPointY - kl2.startPointY;
        float len2 = sqrt(dx2*dx2 + dy2*dy2 + 1e-6f);

        // 3. 计算余弦相似度 (点积 / 长度乘积)
        // cos(theta) = (v1 . v2) / (|v1| * |v2|)
        float cos_theta = (dx1 * dx2 + dy1 * dy2) / (len1 * len2);

        // 4. 由于线段没有方向性（start和end交换也是同一条线），取绝对值
        // 这样即便线段反向了，夹角也会被视为 0 度而不是 180 度
        cos_theta = std::abs(cos_theta);

        // 5. 角度检查
        // 10度对应的余弦值约为 0.9848
        float threshold_cos = cos(angle_threshold_deg * M_PI / 180.0f);

        return cos_theta >= threshold_cos;
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
        if(pKF1->NL==0 || pKF2->NL==0)
            return 0;
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
            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if(dist_12>nn12_dist_th)
            {
                vMatchedPairs.push_back(make_pair(qdx, tdx));
                nmatches++;
            }
        }
        return nmatches;
    }

    void LSDmatcher::SearchForTriangulationLineOld(
        KeyFrame *pKF1, KeyFrame *pKF2,
        vector<pair<int,int>> &vMatchedIdx)
    {
        if(pKF1->NL==0 || pKF2->NL==0)
            return;
        vMatchedIdx.clear();
        //
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


    int LSDmatcher::SearchForTriangulationLine(
        KeyFrame *pKF1, KeyFrame *pKF2,
        vector<pair<int,int>> &vMatchedIdx)
    {
        vMatchedIdx.clear();
        const int N1 = pKF1->NL;
        const int N2 = pKF2->NL;
        if(N1 == 0 || N2 == 0)
            return 0;
        const auto &vL1 = pKF1->mvKeyLines;
        const auto &vL2 = pKF2->mvKeyLines;
        // ========= 1. 计算位姿 =========
        Sophus::SE3f T1w = pKF1->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();
        Sophus::SE3f T21 = T2w * T1w.inverse();   // KF1 → KF2
        Eigen::Matrix3f R21 = T21.rotationMatrix();
        Eigen::Vector3f t21 = T21.translation();
        // ========= 2. 内参 =========
        Eigen::Matrix3f K1 = pKF1->mpCamera->toK_();
        Eigen::Matrix3f K2 = pKF2->mpCamera->toK_();
        Eigen::Matrix3f K1inv = K1.inverse();
        Eigen::Matrix3f K2inv = K2.inverse();
        // ========= 3. KF2 方向预计算 =========
        vector<Eigen::Vector2f> vDir2(N2);
        for(int i=0;i<N2;i++)
        {
            Eigen::Vector2f d(
                vL2[i].endPointX - vL2[i].startPointX,
                vL2[i].endPointY - vL2[i].startPointY);
            vDir2[i] = d.normalized();
        }
        // ========= 4. 加速：Hash Grid =========
        const int GRID = 30;                         // 30x30 栅格
        vector<vector<int>> grid(GRID*GRID);
        float cellX = pKF2->imgLeftRGB.cols / float(GRID);
        float cellY = pKF2->imgLeftRGB.rows / float(GRID);
        auto gridPos = [&](float x, float y){
            int gx = std::min(GRID-1, std::max(0, int(x / cellX)));
            int gy = std::min(GRID-1, std::max(0, int(y / cellY)));
            return gy * GRID + gx;
        };
        for(int i=0;i<N2;i++){
            float mx = (vL2[i].startPointX + vL2[i].endPointX) * 0.5f;
            float my = (vL2[i].startPointY + vL2[i].endPointY) * 0.5f;
            grid[ gridPos(mx,my) ].push_back(i);
        }
        // ========= 旋转直方图 =========
        const int HISTO = 30;
        vector<int> hist[HISTO];
        for(int i=0;i<HISTO;i++) hist[i].reserve(200);
        float factor = 1.f / HISTO;
        // ========= 1:1 互斥匹配 =========
        vector<int> bestFor1(N1, -1);   // KF1 → KF2
        vector<float> bestScore1(N1, 1e9);
        vector<int> bestFor2(N2, -1);   // KF2 → KF1
        vector<float> bestScore2(N2, 1e9);
        // --- 几何阈值 ---
        const float cosAngleTh = cosf(20.f * M_PI / 180.f);
        const float lenRatioTh = 5.0f;
        const float rayTh = 10.0f;
        auto RayResidual = [&](const Eigen::Vector3f &d1,
                           const Eigen::Vector3f &d2)
        {
            Eigen::Vector3f w0 = t21;
            float a = d1.dot(d1);
            float b = d1.dot(d2);
            float c = d2.dot(d2);
            float d = d1.dot(w0);
            float e = d2.dot(w0);
            float denom = a*c - b*b;
            if(fabs(denom) < 1e-6f) return 1e6f;
            float s = (b*e - c*d) / denom;
            float t = (a*e - b*d) / denom;
            Eigen::Vector3f p1 = t21 + s * d1;
            Eigen::Vector3f p2 =       t * d2;
            return (p1 - p2).norm();
        };
        // ========= 主循环：遍历 KF1 所有线条 =========
        for(int i1=0;i1<N1;i1++)
        {
            const auto &L1 = vL1[i1];
            Eigen::Vector2f s1(L1.startPointX, L1.startPointY);
            Eigen::Vector2f e1(L1.endPointX,   L1.endPointY);
            Eigen::Vector2f d1 = (e1 - s1).normalized();
            Eigen::Vector3f r1s = K1inv * Eigen::Vector3f(s1[0], s1[1], 1);
            Eigen::Vector3f r1e = K1inv * Eigen::Vector3f(e1[0], e1[1], 1);
            r1s.normalize(); r1e.normalize();
            // 查找同一区域 grid 中候选 KF2 线
            float mx = (s1[0] + e1[0]) * 0.5f;
            float my = (s1[1] + e1[1]) * 0.5f;
            int cell = gridPos(mx,my);
            const vector<int> &cand = grid[cell];
            if(cand.empty()) continue;
            float len1 = L1.lineLength;
            for(int idx2 : cand)
            {
                const auto &L2 = vL2[idx2];
                Eigen::Vector2f s2(L2.startPointX, L2.startPointY);
                Eigen::Vector2f e2(L2.endPointX,   L2.endPointY);
                // A.方向一致性
                float cosDir = fabs( d1.dot(vDir2[idx2]) );
                if(cosDir < cosAngleTh) continue;
                // B.长度
                float len2 = L2.lineLength;
                float lenDiff = fabs(len1 - len2) / std::max(len1, len2);
                if(lenDiff > lenRatioTh) continue;
                // C.射线几何约束
                Eigen::Vector3f r2s = K2inv * Eigen::Vector3f(s2[0], s2[1], 1);
                Eigen::Vector3f r2e = K2inv * Eigen::Vector3f(e2[0], e2[1], 1);
                r2s.normalize(); r2e.normalize();
                Eigen::Vector3f r1s_2 = R21 * r1s;
                Eigen::Vector3f r1e_2 = R21 * r1e;
                float res = RayResidual(r1s_2, r2s) + RayResidual(r1e_2, r2e);
                if(res > rayTh) continue;
                // 得分更小者取代
                if(res < bestScore1[i1]){
                    bestScore1[i1] = res;
                    bestFor1[i1] = idx2;
                }
                if(res < bestScore2[idx2]){
                    bestScore2[idx2] = res;
                    bestFor2[idx2] = i1;
                }
                // 旋转直方图（用于稍后过滤）
                float angle1 = atan2(d1[1], d1[0]);
                float angle2 = atan2(vDir2[idx2][1], vDir2[idx2][0]);
                float diff = angle1 - angle2;
                diff = diff < 0 ? diff + M_PI*2 : diff;
                int bin = round(diff * factor * HISTO);
                if(bin >= HISTO) bin = 0;
                hist[bin].push_back(i1);
            }
        }
        // ========== 互斥匹配：只有 bestFor1 和 bestFor2 相互同意才保留 ==========
        for(int i1=0;i1<N1;i1++)
        {
            int i2 = bestFor1[i1];
            if(i2 < 0) continue;
            if(bestFor2[i2] == i1){
                vMatchedIdx.emplace_back(i1, i2);
            }
        }
        return vMatchedIdx.size();
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

    // 假设在 LSDmatcher 类内部，必要的 includes 已有：<vector>, <algorithm>, <limits>, <opencv2/opencv.hpp>, Sophus/Eigen 等
int LSDmatcher::SearchForTriangulationLinesRobust(
    KeyFrame *pKF1, KeyFrame *pKF2,
    std::vector<std::pair<int,int>> &vMatchedPairs,
    int K_neighbors /*=5*/, int keepCandidatesPerQuery /*=3*/)
{
    vMatchedPairs.clear();
    if (!pKF1 || !pKF2) return 0;
    if (pKF1->NL == 0 || pKF2->NL == 0) return 0;

    // === 参数（可调） ===
    const float DESC_TH = 100.0f;        // 描述子最大接受距离（Hamming）
    const float MAX_EPIPOLAR = 5.0f;     // 极线平均像素误差阈（较宽松）
    const float MAX_RAY_RES = 5.0f;      // 射线最近点残差阈（像素尺度或归一化尺度）
    const float MAX_LEN_RATIO = 0.7f;    // 长度差容忍（soft）
    const float MAX_ANGLE = 45.0f * M_PI / 180.0f; // 最大角度差（弧度，soft）
    const float EPS = 1e-6f;

    // 权重（用于组合 score；后面会归一化）
    const float w_desc = 1.0f;
    const float w_epi  = 2.0f;
    const float w_ray  = 3.0f;
    const float w_len  = 1.0f;
    const float w_ang  = 1.0f;

    // ==== 预计算 / 准备 ====
    const auto &vL1 = pKF1->mvKeyLines;
    const auto &vL2 = pKF2->mvKeyLines;
    const int N1 = (int)vL1.size();
    const int N2 = (int)vL2.size();

    // 相机内参 & poses
    Eigen::Matrix3f K1 = pKF1->mpCamera->toK_();
    Eigen::Matrix3f K2 = pKF2->mpCamera->toK_();
    Eigen::Matrix3f K1inv = K1.inverse();
    Eigen::Matrix3f K2inv = K2.inverse();

    Sophus::SE3f T1w = pKF1->GetPose();
    Sophus::SE3f T2w = pKF2->GetPose();
    Sophus::SE3f T21 = T2w * T1w.inverse();
    Eigen::Matrix3f R21 = T21.rotationMatrix();
    Eigen::Vector3f t21 = T21.translation();

    // BFMatcher 2-NN 若 K_neighbors>2 改用 knnMatch(k)
    cv::BFMatcher bfm(cv::NORM_HAMMING);

    const cv::Mat &ldesc1 = pKF1->mLineDescriptors;
    const cv::Mat &ldesc2 = pKF2->mLineDescriptors;
    if (ldesc1.empty() || ldesc2.empty()) return 0;

    // knnMatch：对所有 query 做 K_neighbors 最近邻（可能会比较慢）
    std::vector<std::vector<cv::DMatch>> knn;
    bfm.knnMatch(ldesc1, ldesc2, knn, K_neighbors);

    // helper lambdas
    auto endpoint_epipolar_error = [&](const cv::line_descriptor::KeyLine &KLq,
                                       const cv::line_descriptor::KeyLine &KLt)->float
    {
        // 计算 pKF1 两端点映射到 pKF2 极线的平均距离
        Eigen::Vector3f p1s(KLq.startPointX, KLq.startPointY, 1.0f);
        Eigen::Vector3f p1e(KLq.endPointX,   KLq.endPointY,   1.0f);
        // Fundamental matrix F = K2^{-T} * [t]_x * R * K1^{-1}
        Eigen::Matrix3f tx;
        tx <<     0, -t21(2),  t21(1),
               t21(2),     0, -t21(0),
              -t21(1),  t21(0),     0;
        Eigen::Matrix3f F = K2.inverse().transpose() * tx * R21 * K1.inverse();
        Eigen::Vector3f l_s = F * p1s;
        Eigen::Vector3f l_e = F * p1e;
        // distance from a point (x,y) to line l = |ax+by+c|/sqrt(a^2+b^2)
        auto dist_pt_line = [&](const Eigen::Vector3f &l, float x, float y){
            float num = fabs(l(0)*x + l(1)*y + l(2));
            float den = sqrt(l(0)*l(0) + l(1)*l(1)) + EPS;
            return num / den;
        };
        float d_start = dist_pt_line(l_s, KLt.startPointX, KLt.startPointY);
        float d_end   = dist_pt_line(l_e, KLt.endPointX,   KLt.endPointY);
        return 0.5f*(d_start + d_end);
    };

    auto ray_residual_for_pair = [&](const cv::line_descriptor::KeyLine &KLq,
                                     const cv::line_descriptor::KeyLine &KLt)->float
    {
        // 反投影端点到相机射线（单位向量）
        Eigen::Vector3f x1s = K1inv * Eigen::Vector3f(KLq.startPointX, KLq.startPointY, 1.0f);
        Eigen::Vector3f x1e = K1inv * Eigen::Vector3f(KLq.endPointX,   KLq.endPointY,   1.0f);
        Eigen::Vector3f x2s = K2inv * Eigen::Vector3f(KLt.startPointX, KLt.startPointY, 1.0f);
        Eigen::Vector3f x2e = K2inv * Eigen::Vector3f(KLt.endPointX,   KLt.endPointY,   1.0f);
        x1s.normalize(); x1e.normalize(); x2s.normalize(); x2e.normalize();

        // 把 KF1 的射线变换到 KF2 视角： d1' = R21 * d1
        Eigen::Vector3f d1s = R21 * x1s;
        Eigen::Vector3f d1e = R21 * x1e;
        Eigen::Vector3f d2s = x2s;
        Eigen::Vector3f d2e = x2e;

        // 射线最近点残差函数（使用已知 t21）
        auto ray_res = [&](const Eigen::Vector3f &dA, const Eigen::Vector3f &dB)->float
        {
            // Solve for s,t minimizing |t21 + s dA - t dB|
            Eigen::Vector3f w0 = t21;
            float a = dA.dot(dA);
            float b = dA.dot(dB);
            float c = dB.dot(dB);
            float d = dA.dot(w0);
            float e = dB.dot(w0);
            float denom = a*c - b*b;
            if (fabs(denom) < 1e-6f) return 1e6f;
            float s = (b*e - c*d) / denom;
            float t = (a*e - b*d) / denom;
            Eigen::Vector3f p1 = t21 + s * dA;
            Eigen::Vector3f p2 =       t * dB;
            return (p1 - p2).norm();
        };

        float r1 = ray_res(d1s, d2s);
        float r2 = ray_res(d1e, d2e);
        return r1 + r2;
    };

    auto length_ratio = [&](const cv::line_descriptor::KeyLine &KLq,
                            const cv::line_descriptor::KeyLine &KLt)->float
    {
        float l1 = KLq.lineLength;
        float l2 = KLt.lineLength;
        if (l1 <= 0 || l2 <= 0) return 1.0f;
        return fabs(l1 - l2) / std::max(l1, l2); // 0..1
    };

    auto angle_diff = [&](const cv::line_descriptor::KeyLine &KLq,
                          const cv::line_descriptor::KeyLine &KLt)->float
    {
        float a1 = KLq.angle * M_PI / 180.0f;
        float a2 = KLt.angle * M_PI / 180.0f;
        float d = fabs(a1 - a2);
        if (d > M_PI) d = 2*M_PI - d;
        return d; // radians
    };

    // ==== 收集候选并评分 ====
    struct Candidate { int q, t; float score; float descDist; float epi; float ray; float len; float ang; };
    std::vector<Candidate> candidates;
    candidates.reserve(N1 * 4);

    for (int q = 0; q < (int)knn.size(); ++q)
    {
        const auto &kvec = knn[q]; // candidate matches for query q
        if (kvec.empty()) continue;

        // 只处理 top-K_neighbors matches but we'll compute scores for up to K_neighbors
        int upto = std::min((int)kvec.size(), K_neighbors);
        for (int kk = 0; kk < upto; ++kk)
        {
            const cv::DMatch &dm = kvec[kk];
            int t = dm.trainIdx;
            int qidx = dm.queryIdx; // should equal q

            float ddesc = (float)dm.distance;
            if (ddesc > DESC_TH) continue; // descriptor hard threshold

            // 计算几何量
            const auto &KLq = vL1[qidx];
            const auto &KLt = vL2[t];

            float epiErr = endpoint_epipolar_error(KLq, KLt);
            if (epiErr > MAX_EPIPOLAR * 3.0f) continue; // very large epipolar error -> discard early

            float rayRes = ray_residual_for_pair(KLq, KLt);
            // rayRes in normalized units; compare with MAX_RAY_RES
            if (rayRes > MAX_RAY_RES * 5.0f) continue;

            float lenr = length_ratio(KLq, KLt);
            float angd = angle_diff(KLq, KLt);

            // 归一化各项到大致相同范围（heuristic）
            float sn_desc = ddesc / (DESC_TH + EPS);        // 0..1ish
            float sn_epi  = std::min(epiErr / (MAX_EPIPOLAR + EPS), 5.0f);
            float sn_ray  = std::min(rayRes / (MAX_RAY_RES + EPS), 5.0f);
            float sn_len  = std::min(lenr / (MAX_LEN_RATIO + EPS), 5.0f);
            float sn_ang  = std::min(angd / (MAX_ANGLE + EPS), 5.0f);

            float score = w_desc*sn_desc + w_epi*sn_epi + w_ray*sn_ray + w_len*sn_len + w_ang*sn_ang;

            candidates.push_back({qidx, t, score, sn_desc, sn_epi, sn_ray, sn_len, sn_ang});
        }
    }

    if (candidates.empty()) return 0;

    // ==== 只保留每个 q 最好的 keepCandidatesPerQuery 个（减少后续复杂度） ====
    std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b){
        if (a.q != b.q) return a.q < b.q;
        return a.score < b.score;
    });

    std::vector<Candidate> filtered;
    filtered.reserve(candidates.size());
    int last_q = -1, kept = 0;
    for (auto &c : candidates)
    {
        if (c.q != last_q) { last_q = c.q; kept = 0; }
        if (kept < keepCandidatesPerQuery) { filtered.push_back(c); ++kept; }
    }

    // ==== 全局按 score 排序，执行互斥/替换策略（保证 1:1） ====
    std::sort(filtered.begin(), filtered.end(), [](const Candidate &a, const Candidate &b){
        return a.score < b.score;
    });

    // 为互斥匹配维护当前占用与分数（当发现更好分数时可替换）
    std::vector<int> matchQtoT(N1, -1);
    std::vector<float> scoreQ(N1, std::numeric_limits<float>::infinity());
    std::vector<int> matchTtoQ(N2, -1);
    std::vector<float> scoreT(N2, std::numeric_limits<float>::infinity());

    for (const auto &c : filtered)
    {
        int q = c.q;
        int t = c.t;
        float s = c.score;

        // 如果 t 已被其他 q' 占用，只有当当前 s 优于已有分数时替换
        if (matchTtoQ[t] == -1 && matchQtoT[q] == -1)
        {
            matchQtoT[q] = t; scoreQ[q] = s;
            matchTtoQ[t] = q; scoreT[t] = s;
        }
        else if (matchTtoQ[t] != -1 && s < scoreT[t])
        {
            // 替换旧匹配
            int oldq = matchTtoQ[t];
            matchQtoT[oldq] = -1; scoreQ[oldq] = std::numeric_limits<float>::infinity();
            matchQtoT[q] = t; scoreQ[q] = s;
            matchTtoQ[t] = q; scoreT[t] = s;
        }
        else if (matchQtoT[q] != -1 && s < scoreQ[q])
        {
            // q 已匹配到另一个 t0，但当前 s 更好 -> 替换
            int oldt = matchQtoT[q];
            matchTtoQ[oldt] = -1; scoreT[oldt] = std::numeric_limits<float>::infinity();
            matchQtoT[q] = t; scoreQ[q] = s;
            matchTtoQ[t] = q; scoreT[t] = s;
        }
        // 否则跳过
    }

    // ==== 方向直方图过滤（optional, 保留大簇） ====
    if (mbCheckOrientation)
    {
        const int HISTO_LENGTH = 30;
        std::vector<std::vector<int>> rotHist(HISTO_LENGTH);
        float factor = 360.0f / HISTO_LENGTH;
        for (int q = 0; q < N1; ++q)
        {
            int t = matchQtoT[q];
            if (t < 0) continue;
            float a1 = vL1[q].angle;
            float a2 = vL2[t].angle;
            float rot = a1 - a2;
            if (rot < 0.0f) rot += 360.0f;
            int bin = cvRound(rot * factor);
            if (bin == HISTO_LENGTH) bin = 0;
            if (bin >= 0 && bin < HISTO_LENGTH) rotHist[bin].push_back(q);
        }
        int ind1=-1, ind2=-1, ind3=-1;
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
        for (int b = 0; b < HISTO_LENGTH; ++b)
        {
            if (b == ind1 || b == ind2 || b == ind3) continue;
            for (int q : rotHist[b])
            {
                int t = matchQtoT[q];
                if (t >= 0)
                {
                    matchTtoQ[t] = -1;
                    matchQtoT[q] = -1;
                }
            }
        }
    }

    // ==== 填入输出（只返回 q->t 有效的匹配） ====
    int nmatches = 0;
    for (int q = 0; q < N1; ++q)
    {
        int t = matchQtoT[q];
        if (t >= 0)
        {
            // 可选：再做一次严苛重投影检验（CheckLineReprojection），确保最后质量
            // if (!CheckLineReprojection(pKF1, pKF2, ...)) continue;

            vMatchedPairs.emplace_back(q, t);
            ++nmatches;
        }
    }

    return nmatches;
}


    void LSDmatcher::DebugShowTriangulationMatchesKF(
        KeyFrame *pKF1,
        KeyFrame *pKF2,
        const std::vector<std::pair<int,int>> &vMatchedIdx)
    {
        if (!pKF1 || !pKF2) return;
        // -----------------------------
        // 读取图像（优先 RGB，否则转 RGB）
        // -----------------------------
        auto GetRGB = [](const cv::Mat &src)->cv::Mat {
            cv::Mat img;
            if (src.empty()) {
                return cv::Mat(); // 空图像保护
            }
            if (src.channels() == 3)
                img = src.clone();
            else
                cv::cvtColor(src, img, cv::COLOR_GRAY2BGR);
            return img;
        };
        cv::Mat img1 = GetRGB(pKF1->imgLeftRGB);
        cv::Mat img2 = GetRGB(pKF2->imgLeftRGB);
        if (img1.empty() || img2.empty()) return;
        const auto &vLines1 = pKF1->mvKeyLines;
        const auto &vLines2 = pKF2->mvKeyLines;
        // -----------------------------
        // 创建大画布（左右拼接）
        // -----------------------------
        int H = std::max(img1.rows, img2.rows);
        int W = img1.cols + img2.cols;
        cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(0,0,0));
        img1.copyTo(canvas(cv::Rect(0,         0, img1.cols, img1.rows)));
        img2.copyTo(canvas(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
        // -----------------------------
        // 绘制 KeyLine 的辅助函数
        // -----------------------------
        auto DrawLine = [&](const KeyLine &KL, cv::Scalar c, int xoff)
        {
            cv::Point p1(KL.startPointX + xoff, KL.startPointY);
            cv::Point p2(KL.endPointX   + xoff, KL.endPointY);
            cv::line(canvas, p1, p2, c, 2);
        };
        // 所有线段画成灰色
        for (auto &L : vLines1) DrawLine(L, cv::Scalar(100,100,100), 0);
        for (auto &L : vLines2) DrawLine(L, cv::Scalar(100,100,100), img1.cols);
        // -----------------------------
        // 随机颜色
        // -----------------------------
        std::mt19937 rng(0);
        std::uniform_int_distribution<int> uni(0,255);
        // -----------------------------
        // 绘制匹配线段
        // -----------------------------
        for (auto &m : vMatchedIdx)
        {
            int i1 = m.first;
            int i2 = m.second;
            if (i1 < 0 || i1 >= vLines1.size()) continue;
            if (i2 < 0 || i2 >= vLines2.size()) continue;
            cv::Scalar col(uni(rng), uni(rng), uni(rng));
            const KeyLine &L1 = vLines1[i1];
            const KeyLine &L2 = vLines2[i2];
            // 绘制匹配线段
            DrawLine(L1, col, 0);
            DrawLine(L2, col, img1.cols);
            // 中点
            cv::Point mid1((L1.startPointX + L1.endPointX)/2,
                        (L1.startPointY + L1.endPointY)/2);
            cv::Point mid2((L2.startPointX + L2.endPointX)/2 + img1.cols,
                        (L2.startPointY + L2.endPointY)/2);
            // 连线
            cv::line(canvas, mid1, mid2, col, 2);
        }
        // -----------------------------
        // 显示
        // -----------------------------
        cv::imshow("Debug Line Triangulation (with RGB)", canvas);
        cv::waitKey(0);  // 不阻塞
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

    int LSDmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapLine *> &vpLines, float th, vector<MapLine *> &vpReplaceLine)
{
    // =========================================================
    // 1. 安全地从 cv::Mat (CV_32F) 提取 Sim3，并转为 Eigen 矩阵
    // =========================================================
    Eigen::Matrix4f mScw;
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            mScw(i,j) = Scw.at<float>(i,j);
        }
    }
    
    // 提取旋转、平移和尺度
    Eigen::Matrix3f sRcw = mScw.block<3,3>(0,0);
    float s = sRcw.col(0).norm();
    Eigen::Matrix3f Rcw = sRcw / s;
    Eigen::Vector3f tcw = mScw.block<3,1>(0,3) / s;
    Eigen::Vector3f Ow = -Rcw.transpose() * tcw;

    const float fx = pKF->fx;
    const float fy = pKF->fy;
    const float cx = pKF->cx;
    const float cy = pKF->cy;

    int nFused = 0;
    const int nLines = vpLines.size();

    // =========================================================
    // 2. 获取当前帧已有的 MapLine 集合，防止重复匹配
    // =========================================================
    std::vector<MapLine*> vpKFMapLines = pKF->GetMapLineMatches();
    std::unordered_set<MapLine*> spAlreadyFound(vpKFMapLines.begin(), vpKFMapLines.end());
    spAlreadyFound.erase(nullptr);

    // =========================================================
    // 3. 遍历候选 MapLine，投影到该关键帧寻找融合目标
    // =========================================================
    for(int iML = 0; iML < nLines; iML++)
    {
        MapLine* pML = vpLines[iML];

        // 丢弃坏线，或是当前帧已经看到过的线
        if(!pML || pML->isBad() || spAlreadyFound.count(pML))
            continue;

        auto endpoints = pML->GetLineWorldPos();
        Eigen::Vector3f SP = endpoints.first;
        Eigen::Vector3f EP = endpoints.second;

        // 投影到相机坐标系 (注意使用提取出的 Rcw, tcw)
        Eigen::Vector3f SPc = Rcw * SP + tcw;
        Eigen::Vector3f EPc = Rcw * EP + tcw;

        // 深度检查：如果端点在相机后方，跳过该线段
        if (SPc(2) <= 0.0f || EPc(2) <= 0.0f)
            continue;

        // 投影到像素坐标
        float u1 = fx * SPc(0) / SPc(2) + cx;
        float v1 = fy * SPc(1) / SPc(2) + cy;
        float u2 = fx * EPc(0) / EPc(2) + cx;
        float v2 = fy * EPc(1) / EPc(2) + cy;

        // 检查是否在图像范围内
        if (u1 < pKF->mnMinX || u1 > pKF->mnMaxX || v1 < pKF->mnMinY || v1 > pKF->mnMaxY)
            continue;
        if (u2 < pKF->mnMinX || u2 > pKF->mnMaxX || v2 < pKF->mnMinY || v2 > pKF->mnMaxY)
            continue;

        // 计算线段到相机中心的距离，用于预测金字塔层级
        float dist = (0.5f * (SP + EP) - Ow).norm();
        int nPredictedLevel = pML->PredictScale(dist, pKF); 

        // 搜索半径
        float radius = th * pKF->mvScaleFactors[nPredictedLevel];

        // 区域搜索该投影位置附近的线段
        vector<size_t> vIndices = pKF->GetLinesInArea(u1, v1, u2, v2, radius);
        if(vIndices.empty())
            continue;

        cv::Mat dML = pML->GetLineDescriptor();

        int bestDist = 256; 
        int bestIdx = -1;

        // 遍历附近提取到的候选线特征，计算描述子距离
        for(size_t idx : vIndices)
        {
            int klLevel = pKF->mvKeyLines[idx].octave;

            if(klLevel < nPredictedLevel - 1 || klLevel > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mLineDescriptors.row(idx);
            int distDesc = DescriptorDistance(dML, dKF);

            if(distDesc < bestDist)
            {
                bestDist = distDesc;
                bestIdx = idx;
            }
        }

        // 判定匹配成功 (TH_LOW 通常是 50)
        if(bestDist <= TH_LOW)
        {
            MapLine* pMLinKF = pKF->GetMapLine(bestIdx);
            if(pMLinKF)
            {
                // 目标特征已经绑定了一个 MapLine，说明我们找到了物理世界中的同一根线！
                // 把当前帧本地的线特征记录在 vpReplaceLine 数组中
                if(!pMLinKF->isBad())
                {
                    vpReplaceLine[iML] = pMLinKF;
                }
            }
            else
            {
                // 当前帧的这个线特征还没有关联地图线段，直接赋予它
                pML->AddLineObservation(pKF, bestIdx);
                pKF->AddMapLine(pML, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

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

    void LSDmatcher::ComputeThreeMaxima(const std::vector<std::vector<int>> &rotHist,
                        const int HISTO_LENGTH,
                        int &ind1, int &ind2, int &ind3)
    {
        std::vector<std::pair<int,int>> hist;
        hist.reserve(HISTO_LENGTH);
        for(int i = 0; i < HISTO_LENGTH; i++)
        {
            int s = rotHist[i].size();
            hist.emplace_back(s, i);   // <数量, bin 索引>
        }
        // 根据数量排序（从大到小）
        std::sort(hist.begin(), hist.end(),
              [](const std::pair<int,int> &a,
                 const std::pair<int,int> &b){
                    return a.first > b.first;
              });

        ind1 = hist[0].second;
        ind2 = hist[1].second;
        ind3 = hist[2].second;
    }



// 🌟 确保这段代码放在 #if 0 上方，使其能被正常编译！
int LSDmatcher::SearchByProjection(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapLine *> &vpLines, std::vector<MapLine *> &vpMatched, float th)
{
    // =========================================================
    // 1. 安全地从 cv::Mat (CV_32F) 提取 Sim3，并转为 Eigen 矩阵
    // =========================================================
    Eigen::Matrix4f mScw;
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            mScw(i,j) = Scw.at<float>(i,j);
        }
    }
    
    // 提取旋转、平移和尺度
    Eigen::Matrix3f sRcw = mScw.block<3,3>(0,0);
    float s = sRcw.col(0).norm();
    Eigen::Matrix3f Rcw = sRcw / s;
    Eigen::Vector3f tcw = mScw.block<3,1>(0,3) / s;
    Eigen::Vector3f Ow = -Rcw.transpose() * tcw;

    const float fx = pKF->fx;
    const float fy = pKF->fy;
    const float cx = pKF->cx;
    const float cy = pKF->cy;

    // 记录已经匹配过的线段，防止重复匹配
    std::unordered_set<MapLine*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(nullptr);

    int nmatches = 0;
    const int nLines = vpLines.size();

    // =========================================================
    // 2. 遍历候选 MapLine，投影到当前关键帧进行匹配
    // =========================================================
    for(int iML = 0; iML < nLines; iML++)
    {
        MapLine* pML = vpLines[iML];

        // 丢弃坏线和已经匹配的线
        if(!pML || pML->isBad() || spAlreadyFound.count(pML))
            continue;

        auto endpoints = pML->GetLineWorldPos();
        Eigen::Vector3f SP = endpoints.first;
        Eigen::Vector3f EP = endpoints.second;

        // 投影到相机坐标系
        Eigen::Vector3f SPc = Rcw * SP + tcw;
        Eigen::Vector3f EPc = Rcw * EP + tcw;

        // 剔除在相机后方的线段
        if (SPc(2) <= 0.0f || EPc(2) <= 0.0f)
            continue;

        // 投影到像素坐标
        float u1 = fx * SPc(0) / SPc(2) + cx;
        float v1 = fy * SPc(1) / SPc(2) + cy;
        float u2 = fx * EPc(0) / EPc(2) + cx;
        float v2 = fy * EPc(1) / EPc(2) + cy;

        // 图像边界检查
        if (u1 < pKF->mnMinX || u1 > pKF->mnMaxX || v1 < pKF->mnMinY || v1 > pKF->mnMaxY)
            continue;
        if (u2 < pKF->mnMinX || u2 > pKF->mnMaxX || v2 < pKF->mnMinY || v2 > pKF->mnMaxY)
            continue;

        // 计算线段到相机中心的距离，用于预测金字塔层级
        float dist = (0.5f * (SP + EP) - Ow).norm();
        int nPredictedLevel = pML->PredictScale(dist, pKF);

        // 搜索半径
        float radius = th * pKF->mvScaleFactors[nPredictedLevel];

        // 在投影区域内获取候选线段索引
        std::vector<size_t> vIndices = pKF->GetLinesInArea(u1, v1, u2, v2, radius);
        if(vIndices.empty())
            continue;

        cv::Mat dML = pML->GetLineDescriptor();

        int bestDist = 256;
        int bestIdx = -1;

        // 遍历附近提取到的候选线特征，计算描述子距离
        for(size_t idx : vIndices)
        {
            if(vpMatched[idx]) // 如果该检测线段已被其他 MapLine 认领，跳过
                continue;

            int klLevel = pKF->mvKeyLines[idx].octave;

            if(klLevel < nPredictedLevel - 1 || klLevel > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mLineDescriptors.row(idx);
            int distDesc = DescriptorDistance(dML, dKF);

            if(distDesc < bestDist)
            {
                bestDist = distDesc;
                bestIdx = idx;
            }
        }

        // 判断是否匹配成功 (TH_LOW 一般是 50)
        if(bestDist <= TH_LOW)
        {
            vpMatched[bestIdx] = pML;
            nmatches++;
        }
    }

    return nmatches;
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
