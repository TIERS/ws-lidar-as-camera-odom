#ifndef __SUPERPOINTFRONTEND_H_
#define __SUPERPOINTFRONTEND_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core.hpp>
#include <vector>
#include <algorithm>

#include <torch/torch.h>

#include "superpoint/NetWork.h" 


// SuperPointFrontend
class SPFrontend
{
public:
    SPFrontend(std::string weight_path, float nms_dist, float conf_thresh, float nn_thresh);

    std::vector<cv::KeyPoint> nms_fast(torch::Tensor xs, torch::Tensor ys, torch::Tensor prob, int H, int W, float dist_thresh);
    void run(cv::Mat img, std::vector<cv::KeyPoint>& filtered_kpoints, cv::Mat& descMat_out);

private:
    std::shared_ptr<SuperPointNet> model;
    float nms_dist_;
    float conf_thresh_;
    float nn_thresh_;

    int cell_ = 8;
    int border_remove = 4;
    bool cuda = false;
};

#endif