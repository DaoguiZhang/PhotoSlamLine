#include "LSDVisualizer.h"

namespace ORB_SLAM3
{

    cv::Mat LSDVisualizer::DrawLineMatches(const cv::Mat &img1,
                               const std::vector<cv::line_descriptor::KeyLine> &keylines1,
                               const cv::Mat &img2,
                               const std::vector<cv::line_descriptor::KeyLine> &keylines2,
                               const std::vector<cv::DMatch> &matches)
    {
        // 检查输入有效性
        if (img1.empty() || img2.empty()) {
            std::cerr << "[DrawLineMatches] Error: input images are empty." << std::endl;
            return cv::Mat();
        }

        // 创建输出图像，将两张图像水平拼接
        cv::Mat img1_color, img2_color;
        if (img1.channels() == 1)
            cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
        else
            img1.copyTo(img1_color);

        if (img2.channels() == 1)
            cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
        else
            img2.copyTo(img2_color);

        // 拼接两张图像
        cv::Mat outImg;
        cv::hconcat(img1_color, img2_color, outImg);

        const int offset_x = img1.cols; // 第二张图像在输出图像中的x偏移量

        // 为每个匹配绘制线段与连线
        for (const auto &match : matches)
        {
            if (match.queryIdx >= (int)keylines1.size() || match.trainIdx >= (int)keylines2.size())
                continue;

            const cv::line_descriptor::KeyLine &kl1 = keylines1[match.queryIdx];
            const cv::line_descriptor::KeyLine &kl2 = keylines2[match.trainIdx];

            cv::Point2f pt1s(kl1.startPointX, kl1.startPointY);
            cv::Point2f pt1e(kl1.endPointX, kl1.endPointY);

            cv::Point2f pt2s(kl2.startPointX + offset_x, kl2.startPointY);
            cv::Point2f pt2e(kl2.endPointX + offset_x, kl2.endPointY);

            // 生成随机颜色用于每一对匹配
            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);

            // 绘制两张图的线段
            cv::line(outImg, pt1s, pt1e, color, 2);
            cv::line(outImg, pt2s, pt2e, color, 2);

            // 绘制两线段中点之间的连线
            cv::Point2f mid1 = (pt1s + pt1e) * 0.5f;
            cv::Point2f mid2 = (pt2s + pt2e) * 0.5f;
            cv::line(outImg, mid1, mid2, color, 1, cv::LINE_AA);
        }

        return outImg;
    }

} // namespace ORB_SLAM3