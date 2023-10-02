#include "superpoint/PointTracker.hpp"

SingleImage::SingleImage(std::string img_path)
    : img_path_(img_path)
{

}
cv::Mat SingleImage::readImage()
{
    cv::Mat img;
    img = cv::imread(img_path_, cv::IMREAD_GRAYSCALE);
    if (img.data == nullptr)
    {
        std::cerr << "file not exist!" << std::endl;
    }
    img.convertTo(img, CV_32F, 1.0f/255.0f);
    img.copyTo(img_);
    return img;
}
void drawpoints(cv::Mat& img, std::vector<cv::Point2f> outs)
{
    for (auto& out : outs)
    {
        cv::circle(img, out, 2, cv::Scalar(0, 0, 255), -1);
    }
}
void SingleImage::draw(std::vector<cv::KeyPoint>& pts)
{
    std::vector<cv::Point2f> points;
    for (const auto& kpoint : pts)
    {
        const auto& point = kpoint.pt;
        points.push_back(point);
    }
    cv::Mat img_color;
    cv::cvtColor(img_, img_color, cv::COLOR_GRAY2BGR);
    drawpoints(img_color, points);

    cv::imshow("result", img_color);
    cv::imwrite("../export/250.png", img_color*255);
    cv::waitKey(0);

}