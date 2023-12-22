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
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType, int & evaluation_4, double & evaluation_5, double & evaluation_6, double & evaluation_7)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        
        //... TODO : implement FLANN matching
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
 
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1

    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {   // k nearest neighbors (k=2)
        // TODO : implement k-nearest-neighbor matching
        // TODO : filter matches using descriptor distance ratio test


        double t;
        t = (double)cv::getTickCount();

        vector<vector<cv::DMatch>> knn_matches;

        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        double sum_of_distances = 0.0;

        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
                sum_of_distances += (*it)[0].distance;
            }
        }

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        evaluation_6 = 1000 * t / 1.0;

        evaluation_4 = matches.size();



        if (knn_matches.size() > 0 && matches.size() > 0) 
        {
            //Method: Matching Score
            std::vector<cv::Point2f> src_points, dst_points;
            for (const auto& match : matches) 
            {
                src_points.push_back(kPtsSource[match.queryIdx].pt);
                dst_points.push_back(kPtsRef[match.trainIdx].pt);
            }


            //cout << "src_points.size()" <<src_points.size() << endl;///////


            //cout << "dst_points.size():" <<dst_points.size() << endl;///////

            cv::Mat inlier_mask;
            cv::findHomography(src_points, dst_points, cv::RANSAC, 3, inlier_mask);

            int num_inliers = cv::countNonZero(inlier_mask);
            double matching_score = static_cast<double>(num_inliers) / matches.size();


            //Method: Distinctiveness
            double Distinctiveness = static_cast<double>(matches.size()) / knn_matches.size();


            evaluation_5 = Distinctiveness;
            evaluation_7 = matching_score;
        }
        else 
        {
            cout << "knn_matches or matches is zero!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
            evaluation_5=0.0;
            evaluation_7=0.0; 
        }

    }

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


/*


// Detect keypoints in image using the Harris detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img,  std::string which_image, bool bVis)
{


    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);// core function!!!!!
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.

    // STUDENTS NEET TO ENTER THIS CODE (C3.2 Atom 4)

    // Look for prominent corners
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);//计算重叠区域
                    if (kptOverlap > maxOverlap)//true if there is a overlap
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    // visualize keypoints
    if (bVis)
    {
    string windowName = "Harris Corner Detection Results in "+ which_image ;
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(windowName, 7); 
    cv::imshow(windowName, visImage);
    cv::waitKey(1);
    }
    // EOF STUDENT CODE
}

*/


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