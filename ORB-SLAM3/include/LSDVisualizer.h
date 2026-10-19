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
        // Draw matches between two sets of KeyLine features (similar to cv::drawMatches for keypoints)
        static cv::Mat DrawLineMatches(const cv::Mat &img1,
                                       const std::vector<cv::line_descriptor::KeyLine> &keylines1,
                                       const cv::Mat &img2,
                                       const std::vector<cv::line_descriptor::KeyLine> &keylines2,
                                       const std::vector<cv::DMatch> &matches);
        
        cv::Mat DrawLineMatches(const cv::Mat &img1,
                                           const std::vector<cv::line_descriptor::KeyLine> &keylines1,
                                           const cv::Mat &img2,
                                           const std::vector<cv::line_descriptor::KeyLine> &keylines2,
                                           const std::vector<cv::DMatch> &matches,
                                           const std::vector<char> &mask);

        // Draw keylines on a single image. Returns a colored image with lines drawn.
        // color default is green, thickness default is 2.
        //static cv::Mat DrawKeylines(const cv::Mat &img,
        //                            const std::vector<cv::line_descriptor::KeyLine> &keylines,
        //                            const cv::Scalar &color = cv::Scalar(0, 255, 0),
        //                            int thickness = 2);

        static cv::Mat DrawKeylines(const cv::Mat &img,
                                        const std::vector<cv::line_descriptor::KeyLine> &keylines);
    };
} // namespace ORB_SLAM3

#endif //LSDVISUALIZER_H