#ifndef LSDVISUALIZER_H
#define LSDVISUALIZER_H

#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

namespace ORB_SLAM3
{

    class LSDVisualizer {
    public:
        static cv::Mat DrawLineMatches(const cv::Mat &img1,
                                   const std::vector<cv::line_descriptor::KeyLine> &keylines1,
                                   const cv::Mat &img2,
                                   const std::vector<cv::line_descriptor::KeyLine> &keylines2,
                                   const std::vector<cv::DMatch> &matches);
    };
} // namespace ORB_SLAM3

#endif //LSDVISUALIZER_H