/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    //  imgFullFilename csv_output_path newWidth newHeight imgEndIndex
    std::vector<std::string> detectors = {"SURF", "SIFT", "SHITOMASI","HARRIS", "BRISK","FAST", "AKAZE", "ORB"};
    std::vector<std::string> descriptors = {"FREAK", "BRISK", "SURF", "BRIEF",  "AKAZE", "ORB"};

    //I have found that certain combinations are not allowed.
    // when MAT_BF + SEL_KNN, crossCheck has to be false.
    // BRISK, BRIEF, ORB, FREAK, AKAZE:choose DES_BINARY
    // SIFT, SURF: choose DES_HOG
    // AKAZE: descriptors can only be used with KAZE or AKAZE keypoints. 

    // visualize and save options setting
    bool Vis_Keypoints_window = false; // visualize Keypoints resultsctor
    bool bVis_result = true;  // visualize final match results
    bool save_sample = true;  // save final match results
    bool bVis_rubostness = false;  // visualize rubostness results

    // dataset path
    string imgFileType = ".png";
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 1144;   // last file index to load, nubmer of csv lines = imgEndIndex-imgStartIndex

////////////////////////////////////////////////////////////////////////

    for (const std::string& detector : detectors) 
    {
        for (const std::string& descriptor : descriptors) 
        {
            string distanceType = "DES_BINARY"; // DES_BINARY, DES_HOG

            if (((detector.compare("AKAZE")==0 && descriptor.compare ("AKAZE") == 0) || 
                (detector.compare("ORB")==0 && descriptor.compare ("ORB") == 0)     ||
                (detector.compare("AKAZE")!= 0 && descriptor.compare ("AKAZE") != 0 && detector.compare("ORB")!=0 && descriptor.compare ("ORB") != 0)) && 
                !(detector.compare("SURF")==0 && descriptor.compare ("FREAK")==0))
            {
                if (descriptor.compare("SURF")== 0 || descriptor.compare("SIFT") == 0)
                {
                    distanceType = "DES_HOG";
                }

                // misc
                int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
                vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
                // output path
                string csv_output_path = "../csv/" + detector + "_" + descriptor + ".csv";
                // Open  the output file in write mode
                ofstream csvfile(csv_output_path);
                // Write the header (column names) to the CSV file
                csvfile << "image_number,evaluation_1,evaluation_2_1,evaluation_2_2,evaluation_2_3,evaluation_3,evaluation_4,evaluation_5,evaluation_6\n";
                /* MAIN LOOP OVER ALL IMAGES */
                for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
                {
                    /* LOAD IMAGE INTO BUFFER */

                    // assemble filenames for current index
                    ostringstream imgNumber;
                    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;

                    string imgFullFilename = "../../../dataset/signal_image/image_" + imgNumber.str() + imgFileType;

                    // load image from file and convert to grayscale
                    cv::Mat img, imgGray;
                    img = cv::imread(imgFullFilename, cv::IMREAD_GRAYSCALE);

                    int newWidth = 1024;
                    int newHeight = 64;
                    cv::resize(img, imgGray, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

                    //-----------------------------STUDENT ASSIGNMENT 1-----------------------------
                    //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                    // push image into data frame buffer
                    DataFrame frame;
                    frame.cameraImg = imgGray;

                    if (dataBuffer.size() >= dataBufferSize)
                    {
                        dataBuffer.erase(dataBuffer.begin());// the oldest one is deleted from one end of the vector
                    }
                    dataBuffer.push_back(frame);

                    //// EOF STUDENT ASSIGNMENT

                    //cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
                    cout << "------------------------------------" << endl;
                    cout << "Processing image: " << imgIndex << endl;
                    cout << detector + "_" + descriptor << endl;
                    /* DETECT IMAGE KEYPOINTS */

                    // extract 2D keypoints from current image
                    vector<cv::KeyPoint> keypoints; // create empty feature list for current image

                    //-----------------------------STUDENT ASSIGNMENT 2-----------------------------
                    //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detector
                    //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                    string which_image = "original_image"; 

                    double t1 = (double)cv::getTickCount();
                    detKeypoints(keypoints, imgGray, detector, which_image, Vis_Keypoints_window);
                    t1 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();


                    int evaluation_1 = keypoints.size();
                    cout << "Evaluation 1: Number of keypoints detected in single image: "<< keypoints.size() << endl;


                    // --------------optional : limit number of keypoints (helpful for debugging and learning)--------------
                    // because when debugging, we can just check very less keypoints to help us understand.
                    bool bLimitKpts = false;
                    if (bLimitKpts)
                    {
                        int maxKeypoints = 50;//50

                        if (detector.compare("SHITOMASI") == 0)
                        { // for SHITOMASI, there is no response info, so keep the first 50 as they are sorted in descending quality order
                            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                        }
                        cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                        cout << " NOTE: Keypoints have been limited to "<< maxKeypoints << "!" << endl;
                    }

                    // push keypoints and descriptor for current frame to end of data buffer
                    (dataBuffer.end() - 1)->keypoints = keypoints;

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
                    /* EXTRACT KEYPOINT DESCRIPTORS */
                    cv::Mat descriptors;

                    double t2 = (double)cv::getTickCount();
                    descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptor);
                    t2 = ((double)cv::getTickCount() - t2) / cv::getTickFrequency();



                    // push descriptors for current frame to end of data buffer
                    (dataBuffer.end() - 1)->descriptors = descriptors;


            ////////////////////////////////////to compute the Robustness ////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////////////////////////////////////////////

                    // Rotation transformation
                    cv::Mat img_rotated;
                    double angle = 45; // Angle in degrees for image rotation
                    cv::Point2f center(imgGray.cols / 2.0, imgGray.rows / 2.0);
                    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);

                    // Calculate the bounding rectangle size for the rotated image
                    cv::Rect2f bounding_rect = cv::RotatedRect(cv::Point2f(), imgGray.size(), angle).boundingRect2f();

                    // Adjust the rotation matrix to consider the new image center
                    rotation_matrix.at<double>(0, 2) += (bounding_rect.width - imgGray.cols) / 2.0;
                    rotation_matrix.at<double>(1, 2) += (bounding_rect.height - imgGray.rows) / 2.0;

                    // Apply the rotation transformation with the updated rotation matrix and output image size
                    // img_rotated is the output image
                    cv::warpAffine(imgGray, img_rotated, rotation_matrix, bounding_rect.size());

                    //cout << "imgGray.size:" << imgGray.size() << endl;
                    //cout << "img_rotated.size:" << img_rotated.size() <<endl;

                    // detect keypoints in "img_rotated"

                    std::vector<cv::KeyPoint> keypoints_rotated;
                    which_image = "rotated_image"; 
                    detKeypoints(keypoints_rotated, img_rotated, detector, which_image, bVis_rubostness);

                    cv::Mat H = rotation_matrix;
                    H.push_back<double>(cv::Mat::zeros(1, 3, CV_64F));
                    H.at<double>(2, 2) = 1.0;

                    float threshold111 = 4.0;

                    //compute robustness for ratation 
                    double repeatability_rotated = computeRobustness (keypoints, keypoints_rotated, H, threshold111);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

                    // Scaling transformation
                    double scale_factor = 2.0;
                    cv::Mat img_scaled;

                    cv::resize(imgGray, img_scaled, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
                    std::vector<cv::KeyPoint> keypoints_scaled;

                    which_image = "scaled_image"; 

                    detKeypoints(keypoints_scaled, img_scaled, detector, which_image, bVis_rubostness);

                    // Create homography matrix for scaling
                    cv::Mat H_scale = cv::Mat::eye(3, 3, CV_64F);
                    H_scale.at<double>(0, 0) = scale_factor;
                    H_scale.at<double>(1, 1) = scale_factor;
                    H_scale.at<double>(0, 2) = (scale_factor - 1.0) * imgGray.cols / 2.0;
                    H_scale.at<double>(1, 2) = (scale_factor - 1.0) * imgGray.rows / 2.0;

                    double repeatability_scaled = computeRobustness (keypoints, keypoints_scaled, H_scale, threshold111);

            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////////

                    // Blurring transformation
                    int blur_size = 5;
                    cv::Mat img_blurred;
                    cv::GaussianBlur(imgGray, img_blurred, cv::Size(blur_size, blur_size), 0, 0);

                    std::vector<cv::KeyPoint> keypoints_blurred;

                    which_image = "blurred_image"; 

                    detKeypoints(keypoints_blurred, img_blurred, detector, which_image, bVis_rubostness);

                    // Create identity homography matrix for blurring (keypoints positions don't change)
                    cv::Mat H_blur = cv::Mat::eye(3, 3, CV_64F);

                    double repeatability_blurred = computeRobustness (keypoints, keypoints_blurred, H_blur, threshold111);


            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////////

                    double evaluation_2_1 = repeatability_rotated;
                    double evaluation_2_2 = repeatability_scaled;
                    double evaluation_2_3 = repeatability_blurred;

                    cout << "Evaluation 2_1: Robustness of detector (rotation): " << repeatability_rotated<<std::endl;
                    cout << "Evaluation 2_2: Robustness of detector (scaling): " << repeatability_scaled<<std::endl;
                    cout << "Evaluation 2_3: Robustness of detector (blurred): " << repeatability_blurred<<std::endl;

                    /* MATCH KEYPOINT DESCRIPTORS */
                    if (dataBuffer.size() > 1) // wait until at least two images have been processed
                    {
                        vector<cv::DMatch> matches;

                        //-----------------------------STUDENT ASSIGNMENT 5-----------------------------
                        //-----------------------------STUDENT ASSIGNMENT 6-----------------------------
                        //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                        //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp


                        double evaluation_4;
                        double evaluation_5;
                        double evaluation_6; 
                        double t3; 

                        matchDescriptors_and_Distinctiveness((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                        (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                        matches, distanceType, t3, evaluation_4, evaluation_5, evaluation_6);

                        double evaluation_3 = 1000 * (t1+t2+t3)  / 1.0;
                        cout << "Evaluation 3: Computational Efficiency" << evaluation_3 << " ms" << endl;
                        cout << "Evaluation 4: Match Ratio:" << evaluation_4 << endl;
                        cout << "Evaluation 5: Match Score:" << evaluation_5 << std::endl;
                        cout << "Evaluation 6: Distinctiveness:" << evaluation_6 << endl;


                        // Write the image_number and another_value to the CSV file
                        csvfile << imgIndex << "," << evaluation_1  << "," << evaluation_2_1 << "," << evaluation_2_2 << "," << evaluation_2_3 << "," 
                                << evaluation_3 << "," << evaluation_4 << "," << evaluation_5 << "," << evaluation_6 <<"\n";
                        //// EOF STUDENT ASSIGNMENT

                        // store matches in current data frame
                        (dataBuffer.end() - 1)->kptMatches = matches;

                        //cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
                        //cout << "-----------------------------------" << endl;
                        // visualize matches between current and previous image
                        if (bVis_result || (save_sample && imgIndex == 3))
                        {   
                            cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                            cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                            (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                            matches, matchImg,
                                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                                            vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                            //save some final match images to show in the paper 

                            if (bVis_result)
                            {
                                string windowName = "Matching keypoints between two camera images";
                                cv::namedWindow(windowName, 1);    
                                cv::imshow(windowName, matchImg);
                                //cout << "-----------Press key to continue to next image-------------" << endl;
                                cv::waitKey(1); // wait for key to be pressed
                            }

                            
                            if (save_sample && imgIndex == 3)
                            {
                                std::string filename = "../demo_image/out_image_" + detector + "_" + descriptor + ".png";
                                cv::imwrite(filename, matchImg);
                            }
                        }
                    }
                    else
                    {
                        cout << "Still need another image to start the step: Match" << endl;
                    }
                } // eof loop over all images
                // Close the CSV file
                csvfile.close();
                /////////////////////
            } 
        }
    }
    return 0;
}