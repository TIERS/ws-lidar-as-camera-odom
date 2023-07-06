#include <opencv2/opencv.hpp>

#include "superpoint/NetWork.h"
#include "superpoint/PointTracker.h"
#include "superpoint/SuperPointFrontend.h"

int main(int argc, const char *argv[])
{
    std::string img_path = "/home/xianjia/SuperPointCPP/assets/icl_snippet/250.png";
    std::string weight_path = "/home/xianjia/SuperPointCPP/model/superpoint_v3_t.pt";

    // 实例化 SingleImage (将提取的点打印在图片上的类)
    cv::Mat gray_img;
    SingleImage simg(img_path);
    gray_img = simg.readImage();
    // 实例化 SuperPoint前端, 提取特征点
    SPFrontend spfrontend(weight_path, 4, 0.015, 0.7);
    std::vector<cv::KeyPoint> pts;
    cv::Mat descMat_out;
    spfrontend.run(gray_img, pts, descMat_out);

    // 绘图
    simg.draw(pts);
    return 0;
}