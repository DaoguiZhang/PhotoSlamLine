#include"LSDextractor.h"
#include "LSDVisualizer.h"

namespace ORB_SLAM3
{

    int LSDextractor::operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::line_descriptor::KeyLine>& keylines,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea)
    {
        cv::Mat img = _image.getMat();
        ComputeImagePyramid(img);   //compute image pyramid, to do next...
        //detectAndCompute(img, keylines, _descriptors.getMatRef());
        detectAndComputeFiltered(img, keylines, _descriptors.getMatRef());  // 使用过滤后的检测
        int line_number = static_cast<int>(keylines.size());
        return line_number;
    }

    // 检测并提取描述符
    void LSDextractor::detectAndCompute(const cv::Mat& img, std::vector<cv::line_descriptor::KeyLine>& keylines, cv::Mat& descriptors) {
        if(img.empty())
        {
            std::cerr << "LSDextractor: Input image is empty!" << std::endl;
            return;
        }
        cv::Mat gray;
        if (img.channels() == 3)
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else
            gray = img.clone();

        //std::cerr <<"-------------------------" << std::endl;
        //std::cerr << "gray: " << gray.type() << std::endl;
        //std::cerr << "img: " << img.type() << std::endl;
        cv::Mat gray_u;
        gray.convertTo(gray_u, CV_8UC1, 255.0);

        //5, 21: CV_32FC1, CV_32FC3
        // LSD 检测
        std::vector<cv::Vec4f> lines_raw;
        lsd->detect(gray_u, lines_raw);
        //std::cerr << " lsd->detect end: " << std::endl;

        // LLD 描述符
        std::vector<cv::line_descriptor::KeyLine> tempKeylines;
        lld->detect(gray_u, tempKeylines);

        // 剔除短线段
        keylines.clear();
        for (auto &kl : tempKeylines) {
            if (kl.lineLength >= minLineLength)
                keylines.push_back(kl);
        }

        lld->compute(gray_u, keylines, descriptors);

#if 0   //test the line detect
        cv::Mat out_img = LSDVisualizer::DrawKeylines(img, keylines);
        
        // ====== 4. 显示 ======
        cv::imshow("LSD Lines", out_img);
        cv::waitKey(0);
#endif

    }

    void LSDextractor::detectAndComputeFiltered(const cv::Mat& img, std::vector<cv::line_descriptor::KeyLine>& keylines, cv::Mat& descriptors) {
        if(img.empty()) return;

        cv::Mat gray_u;
        if (img.channels() == 3) {
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            gray.convertTo(gray_u, CV_8UC1, 255.0);
        } else {
            img.convertTo(gray_u, CV_8UC1, 255.0);
        }

        // 1. 原始检测
        std::vector<cv::line_descriptor::KeyLine> allKeylines;
        lld->detect(gray_u, allKeylines);

        // 2. 针对百叶窗：执行基于网格的非极大值抑制 (NMS)
        // 目的：在局部密集区域只保留最具有代表性的长线段
        int gridSize = mGridSize; // 网格大小可以根据分辨率调整
        int nCols = gray_u.cols / gridSize + 1;
        int nRows = gray_u.rows / gridSize + 1;
    
        // 创建一个容器，存储每个网格内质量最好的线段
        // 这里以网格中心点所在的 bucket 存储
        std::vector<std::vector<cv::line_descriptor::KeyLine>> gridBuckets(nCols * nRows);

        for (auto &kl : allKeylines) {
            // A. 基础过滤：剔除太短的（百叶窗区域会有大量碎线）
            if (kl.lineLength < minLineLength) continue;

            // B. 计算中心点决定网格归属
            int cellX = static_cast<int>(kl.pt.x) / gridSize;
            int cellY = static_cast<int>(kl.pt.y) / gridSize;
            gridBuckets[cellY * nCols + cellX].push_back(kl);
        }

        keylines.clear();
        const size_t maxLinesPerCell = 3; // 每个网格最多保留3条线（针对百叶窗关键设置）

        for (auto &bucket : gridBuckets) {
            if (bucket.empty()) continue;

            // 按照长度排序，优先保留长线
            std::sort(bucket.begin(), bucket.end(), [](const cv::line_descriptor::KeyLine& a, const cv::line_descriptor::KeyLine& b) {
                return a.lineLength > b.lineLength;
            });

            // 取前 N 条
            for (size_t i = 0; i < std::min(bucket.size(), maxLinesPerCell); ++i) {
                keylines.push_back(bucket[i]);
            }
        }

        // 3. 计算描述符（只针对筛选后的精简线段，效率大大提升）
        if (!keylines.empty()) {
            lld->compute(gray_u, keylines, descriptors);
        }

        // [DEBUG] 打印一下保留下来的线段数量，看看百叶窗区域是否还爆炸
        std::cerr << "LSD: Raw=" << allKeylines.size() << " -> Filtered=" << keylines.size() << std::endl;
    }

    void LSDextractor::ComputeImagePyramid(const cv::Mat& image)
    {
        mvImagePyramid.resize(nlevels);
        mvImagePyramid[0] = image.clone();
        for (int level = 1; level < nlevels; ++level) {
            float scale = mvInvScaleFactor[level];
            cv::Size sz(cvRound(image.cols * scale), cvRound(image.rows * scale));
            cv::resize(image, mvImagePyramid[level], sz, 0, 0, cv::INTER_AREA);
        }
    }

}
