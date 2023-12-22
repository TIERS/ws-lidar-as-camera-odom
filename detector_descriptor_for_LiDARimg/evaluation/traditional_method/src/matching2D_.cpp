#include <numeric>
#include <typeinfo>
#include "matching2D.hpp"

using namespace std;

double computeRobustness (const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const cv::Mat& H, float threshold ) 
{
    std::vector<cv::Point2f> points1, points2;

    for (const auto& kp : keypoints1) {
        points1.push_back(kp.pt);
    }
    for (const auto& kp : keypoints2) {
        points2.push_back(kp.pt);
    }

    std::vector<cv::Point2f> transformed_points1;
    cv::perspectiveTransform(points1, transformed_points1, H);

    // cout << "transformed_points1:"<<transformed_points1.size() << endl;
    int count = 0;

    for (const auto& transformed_point : transformed_points1) 
    {
        float min_dist = std::numeric_limits<float>::max();
        for (const auto& point2 : points2) 
        {
            float dist = cv::norm(transformed_point - point2);
            min_dist = std::min(min_dist, dist);
        }
        if (min_dist < threshold) {
            count++;
        }
    }

    // std::cout << "count:"<<count << std::endl;
    return static_cast<double>(count) / keypoints1.size();
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors_and_Distinctiveness(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string distanceType, double & t3, double & evaluation_4, double & evaluation_5, double & evaluation_6)

{ 

    // configure matcher
    bool crossCheck = true;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    int normType = distanceType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
    matcher = cv::BFMatcher::create(normType, crossCheck);
    double t;
    t = (double)cv::getTickCount();
    matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    t3 = ((double)cv::getTickCount() - t) / cv::getTickFrequency();


    if ( matches.size() > 0) 
    {
        double match_ratio = static_cast<double>(matches.size()) / kPtsRef.size();
        evaluation_4 = match_ratio;

        ////////////////////////////////////////////////
        //Method: Match Score
        std::vector<cv::Point2f> src_points, dst_points;
        for (const auto& match : matches) 
        {
            src_points.push_back(kPtsSource[match.queryIdx].pt);
            dst_points.push_back(kPtsRef[match.trainIdx].pt);
        }
        cv::Mat inlier_mask;
        cv::findHomography(src_points, dst_points, cv::RANSAC, 3, inlier_mask);

        int num_inliers = cv::countNonZero(inlier_mask);
        double match_score = static_cast<double>(num_inliers) / matches.size();
        evaluation_5 = match_score;
        ////////////////////////////////////////////////
    }
    else 
    {
        cout << "Match is zero!!"<<endl;
        evaluation_4=0.0;
        evaluation_5=0.0; 
    }
        

    //////////////////////////////////////////////////////
    //Method: Distinctiveness
    cv::Ptr<cv::DescriptorMatcher> knn_matcher;
    knn_matcher = cv::BFMatcher::create(normType, false);

    vector<vector<cv::DMatch>> knn_matches;
    vector<cv::DMatch> knn_matches_final;

    knn_matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches

    // filter matches using descriptor distance ratio test
    double minDescDistRatio = 0.8;
    double sum_of_distances = 0.0;

    for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
    {
        if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
        {
            knn_matches_final.push_back((*it)[0]);
            sum_of_distances += (*it)[0].distance;
        }
    }

    if (knn_matches.size() < 1 ) 
    {
        evaluation_6=0.0;
        cout << "Knn_matches for Distinctiveness is zero!!"<<endl;
    }
    else if (knn_matches_final.size() < 1)
    {
        evaluation_6=0.0;
        cout << "Knn_matches_final for Distinctiveness is zero!!"<<endl;
    }
    else 
    {
        double Distinctiveness = static_cast<double>(knn_matches_final.size()) / knn_matches.size();
        evaluation_6 = Distinctiveness;
    }

    //////////////////////////////////////////////////////
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT 
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    }
    else if (descriptorType.compare("SURF") == 0)
    {
        extractor = cv::xfeatures2d::SURF::create();
    }
    else
    {
        cerr << "Parameter: descriptorType, is wrong. Only one of these, is allowed: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT" << endl;
        exit(1);
    }

    // perform feature description
    extractor->compute(img, keypoints, descriptors);

}

void detKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, std::string which_image, bool bVis)
{
    if (detectorType.compare("SHITOMASI") == 0)
    {
        detKeypointsShiTomasi(keypoints, img, which_image, bVis);
    }
    else if (detectorType.compare("HARRIS") == 0)
    {
        detKeypointsHarris(keypoints, img, which_image, bVis);
    }
    else if ((detectorType.compare("FAST") == 0) || (detectorType.compare("BRISK") == 0) || (detectorType.compare("ORB") == 0) || (detectorType.compare("AKAZE") == 0) || (detectorType.compare("SIFT") == 0) || (detectorType.compare("SURF") == 0))
    {
        detKeypointsModern(keypoints, img, detectorType,which_image, bVis);
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string which_image, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection

    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results in " + which_image;
        cv::namedWindow(windowName, 7);
        imshow(windowName, visImage);
        cv::waitKey(1);
    }
}

// Detect keypoints in image using the Harris detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img,  std::string which_image, bool bVis)
{

    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection

    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, true, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results in " + which_image;
        cv::namedWindow(windowName, 7);
        imshow(windowName, visImage);
        cv::waitKey(1);
    }
}


// Detect keypoints in image using the modern detectors
void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, std::string which_image, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);

    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();

    }
    else if (detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
        //cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
    }
    else if (detectorType.compare("SURF") == 0)
    {
        detector = cv::xfeatures2d::SURF::create();

    }
    detector->detect(img, keypoints);
    // visualize results
    if (bVis)
    {       
        string windowName = detectorType + " Detector Results in " + which_image;
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 7);
        imshow(windowName, visImage);
        cv::waitKey(1);
    }
}