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
    std::vector<std::string> detectors = {"HARRIS","SIFT", "SURF","ORB","AKAZE","FAST","SHITOMASI",};
    std::vector<std::string> descriptors = {"SURF", "BRIEF", "BRISK", "AKAZE", "ORB", "SIFT", "SIFT"};

    string matcherType = "MAT_BF"; // MAT_BF, MAT_FLANN
    string selectorType = "SEL_KNN"; // SEL_NN, SEL_KNN

    //I have found that certain combinations are not allowed.
    // when MAT_BF + SEL_KNN, crossCheck has to be false.
    // BRISK, BRIEF, ORB, FREAK, AKAZE:choose DES_BINARY
    // SIFT, SURF: choose DES_HOG
    // AKAZE: descriptors can only be used with KAZE or AKAZE keypoints. 

    // visualize options setting
    bool Vis_Keypoints_window = false; // visualize Keypoints results
    bool bVis = false;  // visualize final match results
    bool bVis_rubostness = false;  // visualize final match results
    bool save_sample = false;  // save sample keypoint results

    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 1145;   // last file index to load


    for (const std::string& detector : detectors) 
    {
        for (const std::string& descriptor : descriptors) 
        {
            std::string detectorType = "nothing";
            std::string descriptorType = "nothing";
            string distanceType = "DES_BINARY"; // DES_BINARY, DES_HOG

            if ( ((detector.compare("AKAZE")==0 && descriptor.compare ("AKAZE") == 0) || (detector.compare("ORB")==0 && descriptor.compare ("ORB") == 0)) || ((detector.compare("AKAZE")!= 0 && descriptor.compare ("AKAZE") != 0) && (detector.compare("ORB")!=0 && descriptor.compare ("ORB") != 0)) )
            {
                detectorType = detector;
                descriptorType = descriptor;
            }

            if (detectorType.compare("nothing")!= 0 && descriptorType.compare("nothing")!= 0)
            {
                if (descriptorType.compare("SURF")== 0 || descriptorType.compare("SIFT") == 0)
                {
                    distanceType = "DES_HOG";
                }

                // Define an array of interpolation methods
                int interpolation_methods[] = {cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::INTER_LANCZOS4, cv::INTER_AREA};
                std::string interpolation_names[] = {"NEAREST", "LINEAR", "CUBIC", "LANCZOS4", "AREA"};

                //////////////////////////////
                //////////////////////////////
                // Loop over desired dimensions
                for(int width =512; width <=4096; width += 128)//2048
                {
                    for(int height = 32; height <= 256; height += 32)//128//for(int height = 32; height <= 256; height += 32)
                    {
                        for(int i = 0; i < 5; i++)
                        {

                            // misc
                            int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
                            vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

                            string output_path = "../csv_output/" + detectorType + "_" + descriptorType + "_" + std::to_string(width) + "x" + std::to_string(height) + "_" + interpolation_names[i] + ".csv";
                            // Open  the output file in write mode
                            ofstream csvfile(output_path);
                            // Write the header (column names) to the CSV file
                            csvfile << "image_number,evaluation_1,evaluation_2_1,evaluation_2_2,evaluation_2_3,evaluation_3_1,evaluation_3_2,evaluation_3_3,evaluation_4,evaluation_5,evaluation_6,evaluation_7\n";
      
                            // loop images in every 10 images
                            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += 10)
                            {
                                /* LOAD IMAGE INTO BUFFER */
      
                                // assemble filenames for current index
                                ostringstream imgNumber;
                                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                                string imgFullFilename = "../signal_image/image_" + imgNumber.str() + ".png";

                                // load image from file and convert to grayscale
                                cv::Mat img, imgGray;
                                img = cv::imread(imgFullFilename, cv::IMREAD_GRAYSCALE);
                                cv::resize(img, imgGray, cv::Size(width, height), 0, 0, interpolation_methods[i]);

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
                                cout << "Processing image: " << imgIndex << output_path<< endl;
                                /* DETECT IMAGE KEYPOINTS */

                                // extract 2D keypoints from current image
                                vector<cv::KeyPoint> keypoints; // create empty feature list for current image

                                //-----------------------------STUDENT ASSIGNMENT 2-----------------------------
                                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                                string which_image = "original_image"; 

                                double t1 = (double)cv::getTickCount();

                                detKeypoints(keypoints, imgGray, detectorType, which_image, Vis_Keypoints_window);



                                t1 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
                                int evaluation_1 = keypoints.size() ;
                                double evaluation_2_1 = 1000 * t1 / 1.0;
                                cout << "Evaluation 1: number of keypoints detected in this image: "<< keypoints.size() << endl;
                                cout << "Evaluation 2.1: Time taken by detectors : "<< 1000 * t1 / 1.0 << " ms" << endl;



                                if ( evaluation_1 == 0)
                                {       
                                    cout << "no keypoint detection in this image???????????????????????????????????????????????????????????????????????"<< endl;
                                }                          


                                //cout << "#2 : DETECT KEYPOINTS done" << endl;
                                //cout << "-----------------------------------" << endl;

                                // --------------optional : limit number of keypoints (helpful for debugging and learning)--------------
                                // because when debugging, we can just check less keypoints to help us understand.
                                bool bLimitKpts = false;
                                if (bLimitKpts)
                                {
                                    int maxKeypoints = 50;//50

                                    if (detectorType.compare("SHITOMASI") == 0)
                                    { // for SHITOMASI, there is no response info, so keep the first 50 as they are sorted in descending quality order
                                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                                    }
                                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                                    cout << " NOTE: Keypoints have been limited to 10!" << endl;
                                }

                                // push keypoints and descriptor for current frame to end of data buffer
                                (dataBuffer.end() - 1)->keypoints = keypoints;

                        ////////////////////////////////////////////////////////////////////////////////////////////////////////
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////
                                /* EXTRACT KEYPOINT DESCRIPTORS */
                                cv::Mat descriptors;

                                double t2 = (double)cv::getTickCount();
                                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
                                t2 = ((double)cv::getTickCount() - t2) / cv::getTickFrequency();

                                double evaluation_2_2 = 1000 * t2 / 1.0;
                                double evaluation_2_3 = 1000 * (t1+t2)  / 1.0;
                                cout << "Evaluation 2.2: Time taken by descriptors: " << 1000 * t2 / 1.0 << " ms" << endl;
                                cout << "Evaluation 2.3: Time taken by detectors and descriptors: "<< 1000 * (t1+t2) / 1.0 << " ms" << endl;
                                //// EOF STUDENT ASSIGNMENT

                                // push descriptors for current frame to end of data buffer
                                (dataBuffer.end() - 1)->descriptors = descriptors;

                                //cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
                                //cout << "-----------------------------------" << endl;
                        ////////////////////////////////////to compute the Robustness ////////////////////////////////////////////
                        //////////////////////////////////////////////////////////////////////////////////////////////////////////

                                double evaluation_3_1 = 0.0;
                                double evaluation_3_2 = 0.0;
                                double evaluation_3_3 = 0.0;


                                if (evaluation_1 > 0)
                                {
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
                                    detKeypoints(keypoints_rotated, img_rotated, detectorType, which_image, bVis_rubostness);

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

                                    detKeypoints(keypoints_scaled, img_scaled, detectorType, which_image, bVis_rubostness);

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

                                    detKeypoints(keypoints_blurred, img_blurred, detectorType, which_image, bVis_rubostness);

                                    // Create identity homography matrix for blurring (keypoints positions don't change)
                                    cv::Mat H_blur = cv::Mat::eye(3, 3, CV_64F);

                                    double repeatability_blurred = computeRobustness (keypoints, keypoints_blurred, H_blur, threshold111);


                                    evaluation_3_1 = repeatability_rotated;
                                    evaluation_3_2 = repeatability_scaled;
                                    evaluation_3_3 = repeatability_blurred;

                                    cout << "Evaluation 3.1: robustness (rotation): " << repeatability_rotated*100 <<"%"<< std::endl;
                                    cout << "Evaluation 3.2: robustness (scaling): " << repeatability_scaled*100 <<"%"<< std::endl;
                                    cout << "Evaluation 3.3: robustness (blurred): " << repeatability_blurred*100 <<"%"<< std::endl;
                                }

                        ///////////////////////////////////////////////////////////////////////////////////////////////////////
                        ///////////////////////////////////////////////////////////////////////////////////////////////////////


                                /* MATCH KEYPOINT DESCRIPTORS */
                                if (dataBuffer.size() > 1 && evaluation_1 > 0 ) // wait until at least two images have been processed
                                {
                                    vector<cv::DMatch> matches;

                                    //-----------------------------STUDENT ASSIGNMENT 5-----------------------------
                                    //-----------------------------STUDENT ASSIGNMENT 6-----------------------------
                                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
                                    
                                    int    evaluation_4;
                                    double evaluation_5;
                                    double evaluation_6; 
                                    double evaluation_7;

                                    matchDescriptors_and_Distinctiveness((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                                    matches, distanceType, matcherType, selectorType, evaluation_4, evaluation_5, evaluation_6, evaluation_7);

                                    cout << "Evaluation 4: Number of matches between consecutive images:" <<evaluation_4 << endl;
                                    cout << "Evaluation 5: Distinctiveness of detector: " << evaluation_5 << endl;
                                    cout << "Evaluation 6: Time taken by matching process:  " << evaluation_6 << " ms" << endl;
                                    cout << "Evaluation 7: Matching Score: " << evaluation_7 << std::endl;

                                    // Write the image_number and another_value to the CSV file
                                    csvfile << imgIndex << "," << evaluation_1  << "," << evaluation_2_1 << "," << evaluation_2_2 << "," << evaluation_2_3 << "," 
                                            << evaluation_3_1 << "," << evaluation_3_2 << "," << evaluation_3_3 << "," 
                                            << evaluation_4 << "," << evaluation_5 << "," << evaluation_6 << "," << evaluation_7 << "\n";
                                    //// EOF STUDENT ASSIGNMENT

                                    // store matches in current data frame
                                    (dataBuffer.end() - 1)->kptMatches = matches;

                                    //cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
                                    //cout << "-----------------------------------" << endl;
                                    // visualize matches between current and previous image
                                    if (bVis)
                                    {
                                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                                        matches, matchImg,
                                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                                        string windowName = "Matching keypoints between two camera images";
                                        cv::namedWindow(windowName, 1);    
                                        cv::imshow(windowName, matchImg);
                                        //std::string filename = "../demo_image_of_traditional_method/out_image" + std::to_string(imgIndex) + ".png";
                                        //cv::imwrite(filename, matchImg);  //save some match sample image for our paper
                                        cv::waitKey(1); 
                                    }
                                }
                                else
                                {
                                    cout << "Still need another image to start the step: Match" << endl;
                                }
                            } // eof loop over all images

                            // Close the CSV file
                            csvfile.close();

                        }
                    }
                }

                /////////////////////
                /////////////////////

            }

        }
    }

    return 0;
}
