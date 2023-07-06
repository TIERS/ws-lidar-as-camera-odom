#ifndef __POINTTRACKER_HPP_
#define __POINTTRACKER_HPP_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core.hpp>

class SingleImage
{
public:
    SingleImage(std::string img_path);
    cv::Mat readImage();
    void draw(std::vector<cv::KeyPoint>& pts);

private:
    std::string img_path_;
    cv::Mat img_;
};

#endif