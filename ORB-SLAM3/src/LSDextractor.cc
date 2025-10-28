#include"LSDextractor.h"

namespace ORB_SLAM3
{

    int LSDextractor::operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::line_descriptor::KeyLine>& keylines,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea)
    {
        cv::Mat img = _image.getMat();
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
            cvtColor(img, gray, cv::COLOR_BGR2GRAY);
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
}