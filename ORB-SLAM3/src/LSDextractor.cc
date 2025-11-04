#include"LSDextractor.h"

namespace ORB_SLAM3
{

    int LSDextractor::operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::line_descriptor::KeyLine>& keylines,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea)
    {
        cv::Mat img = _image.getMat();
        ComputeImagePyramid(img);   //compute image pyramid, to do next...
        detectAndCompute(img, keylines, _descriptors.getMatRef());
        return 0;
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

        // LSD 检测
        std::vector<cv::Vec4f> lines_raw;
        lsd->detect(gray, lines_raw);

        // LLD 描述符
        std::vector<cv::line_descriptor::KeyLine> tempKeylines;
        lld->detect(gray, tempKeylines);

        // 剔除短线段
        keylines.clear();
        for (auto &kl : tempKeylines) {
            if (kl.lineLength >= minLineLength)
                keylines.push_back(kl);
        }

        lld->compute(gray, keylines, descriptors);
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