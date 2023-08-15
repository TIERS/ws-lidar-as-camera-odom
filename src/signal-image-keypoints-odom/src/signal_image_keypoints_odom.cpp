#include "signal_image_keypoints_odom/signal_image_keypoints_odom.hpp"


lidarImageKeypointOdom::lidarImageKeypointOdom(ros::NodeHandle *nh_)
{
    pointCloudSub_ = nh_->subscribe<sensor_msgs::PointCloud2>("/os0_cloud_node/points", 10, &lidarImageKeypointOdom::pointCloudCallback, this);

    keyPointsSub_ = nh_->subscribe<sensor_msgs::PointCloud2>("/for_doctor_yu", 10, &lidarImageKeypointOdom::keyPointsCallback, this);

    signalImagesSub_ = nh_->subscribe<sensor_msgs::Image>("/os0_img_node/signal_image", 10, &lidarImageKeypointOdom::signalImageCallback, this);

    keyPointCloudPub_ =  nh_->advertise<sensor_msgs::PointCloud2>("/keypoint_point_cloud", 1);

    timer_ = nh_->createTimer(ros::Duration(0.1), &lidarImageKeypointOdom::superpointtimerCallback, this);

    imu_sub_ = nh_->subscribe("/imu_topic", 10, &lidarImageKeypointOdom::imuCallback, this);

    imu_pub_ = nh_->advertise<sensor_msgs::Imu>("/keypoint_imu", 10);
    


    ROS_INFO("OpenCV version: %s", cv::getVersionString().c_str());


#ifdef sensor_tpye=="os0"
    for (int i = 0; i < SH_; i++)  {
        pixel_shift_by_row_[i] = initial_arr[i%4];
    }
#else
    for (int i = 0; i < SH_; i++){
        pixel_shift_by_row_[i] = initial_arr[i%2];
    }
#endif
}

lidarImageKeypointOdom::~lidarImageKeypointOdom()
{
}

void lidarImageKeypointOdom::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg){
    // ROS_INFO(">>>>>>>> camera pointcloud >>>>>>>>>");
    pcl::PCLPointCloud2 pcl_pc2;

    pcl_conversions::toPCL(*msg,pcl_pc2);


    pcl::fromPCLPointCloud2(pcl_pc2,*entirePointCloudPtr_);

    pc_time_ = std::chrono::steady_clock::now();
    ROS_INFO(">>>>>>>> camera pointcloud >>>>>>>>>");
}


void lidarImageKeypointOdom::imuCallback(const sensor_msgs::ImuConstPtr& imu) {
        // Store the received IMU data
        last_received_imu_ = *imu;  // Directly assign
    }


void lidarImageKeypointOdom::keyPointsCallback(const sensor_msgs::PointCloud2::ConstPtr& msg){
    pcl::PCLPointCloud2 pcl_pc2;

    pcl_conversions::toPCL(*msg,pcl_pc2);

    pcl::fromPCLPointCloud2(pcl_pc2,*keyPointsPtr_);
}

void lidarImageKeypointOdom::signalImageCallback(const sensor_msgs::ImageConstPtr& msg){ 
    try
    {
       cvPtr_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO16);
       img_ = cvPtr_->image;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    img_time_ = std::chrono::steady_clock::now();
    ROS_INFO(">>>>>>>> sigal images received!!! >>>>>>>>>");
}



std::vector<cv::KeyPoint> extract_neighbourhood(const cv::Mat& img, const cv::Point& point, int size = 9, int threshold = 150) {
    std::vector<cv::KeyPoint> neibor;
    int half_size = size / 2;
    int x_min = std::max(0, point.x - half_size);
    int x_max = std::min(img.cols, point.x + half_size + 1);
    int y_min = std::max(0, point.y - half_size);
    int y_max = std::min(img.rows, point.y + half_size + 1);

    cv::Mat neighbourhood = img(cv::Range(y_min, y_max), cv::Range(x_min, x_max));
    cv::Mat diff_mask = cv::abs(neighbourhood - img.at<uchar>(point.y, point.x)) < threshold;

    for (int y = 0; y < diff_mask.rows; ++y) {
        for (int x = 0; x < diff_mask.cols; ++x) {
            if (diff_mask.at<uchar>(y, x)) {
                cv::KeyPoint kp(point.x - half_size + x, point.y - half_size + y, 1);  // Size is set to 1 for the KeyPoint, adjust if needed
                neibor.push_back(kp);
            }
        }
    }
    return neibor;
}
 
void lidarImageKeypointOdom::superpointtimerCallback(const ros::TimerEvent& event){

    if(img_.empty() | entirePointCloudPtr_->empty()){
        ROS_WARN("Either no image or no point cloud input!");
        return ;
    }
    std::chrono::steady_clock::time_point _start(std::chrono::steady_clock::now());
    float img_gap = std::chrono::duration_cast<std::chrono::duration<double>>(img_time_ - _start).count();
    float pc_gap = std::chrono::duration_cast<std::chrono::duration<double>>(pc_time_ - _start).count();
    if(std::fabs(pc_gap) > 0.2 | std::fabs(img_gap) > 0.2){
        {
        ROS_WARN("No input data!");
        return ;
    }
    }
    cv::Mat mp = img_.clone();

    // img_queue_.push(mp);
    // if(img_queue_.size() < 2){
    //     ROS_WARN("Not enough image to match!");
    //     return;
    // }

    // cv::Mat img0 = img_queue_.front();
    // img_queue_.pop();
    // cv::Mat img1 = img_queue_.front();

    // cv::Mat resized0, resized1, img_gray0, img_gray1;
    // cv::resize(img0, resized0, cv::Size(SW_/2, SH_/2), cv::INTER_AREA);
    // cv::resize(img1, resized1, cv::Size(SW_/2, SH_/2), cv::INTER_AREA);
    // cv::cvtColor(resized0, img_gray0, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(resized1, img_gray1, cv::COLOR_BGR2GRAY);



    mp.convertTo(mp, CV_8U, 1.0 / 256.0);

    // Resize the image
    cv::resize(mp, mp, cv::Size(), 0.5, 0.5, cv::INTER_CUBIC);

    // Convert to float32 in range [0,1]
    mp.convertTo(mp, CV_32F, 1.0 / 255.0);




    // cv::Mat resized, img_gray;;
    // cv::resize(mp, resized, cv::Size(SW_/2, SH_/2), cv::INTER_AREA);
    // cv::cvtColor(resized, img_gray, cv::COLOR_BGR2GRAY);
    // cv::imwrite("/home/xianjia/img_gray.png", img_gray);
    std::vector<cv::KeyPoint> pts;
    cv::Mat descMat_out;
    spfrontend_.run(mp, pts, descMat_out);
    ROS_INFO("superpoints size: %d", pts.size());
    

    // if(img_gray0.empty() || img_gray1.empty()){
    //     return;
    // }   
    // cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    // std::vector<cv::KeyPoint> kp0, kp1;
    // std::vector<cv::DMatch> matches;
    // cv::Mat des0, des1;

    // detector->detectAndCompute(img0, cv::Mat(), kp0, des0);
    // detector->detectAndCompute(img1, cv::Mat(), kp1, des1);

    // matcher->match(des1, des0, matches);


    // ROS_INFO("kp0 number: %d, kp1 number: %d,matches number: %d", kp0.size(), kp1.size(), matches.size());

    // if(kp0.empty() || kp1.empty() ){
    //     ROS_WARN("No keypoints detected!");
    //     return;
    // }
   
    // std::sort(matches.begin(), matches.end());
    // const int match_size = matches.size();

    // std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + (int)(match_size * 0.9f));

    std::vector<cv::KeyPoint> neibor;
    for (const auto& keypoint : pts) {
        // std::cout << keypoint.pt.x << ", " << keypoint.pt.y << std::endl;
        std::vector<cv::KeyPoint> neighbourhood = extract_neighbourhood(mp, keypoint.pt, 6, 300);
        neibor.insert(neibor.end(), neighbourhood.begin(), neighbourhood.end());
        std::cout << neibor.size() << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloudPtr{new pcl::PointCloud<pcl::PointXYZ>};

    // // cv::Mat output;
    // // std::string output_name = "/home/xianjia/img-" + std::to_string(cnt_)+".jpg";
    // // std::cout << output_name << std::endl;
    // // cv::drawKeypoints(mp, keypoints,output);
    // // cv::imwrite(output_name, output);
    // cnt_ = cnt_+ 1;

    
    keyPointCloudPtr->header = entirePointCloudPtr_->header;
    keyPointCloudPtr->width = entirePointCloudPtr_->width;
    keyPointCloudPtr->height = entirePointCloudPtr_->height;
    keyPointCloudPtr->points.resize(keyPointCloudPtr->width * keyPointCloudPtr->height);
    // keyPointCloudPtr->clear();
    
    // std::cout << keyPointCloudPtr->size() << std::endl;
    for(auto &gm : neibor){
        size_t u = (size_t) gm.pt.y * 2;
        size_t v = (size_t) gm.pt.x * 2;
        if (u>SH_ | v > SW_)
            continue;
        if (u == keyPointCloudPtr->height)
        {
            u--;
        }
        if (v == keyPointCloudPtr->width)
        {
            v--;
        }
        
        // std::cout << u << " " << v << std::endl;
        keyPointCloudPtr->at(v,u) = entirePointCloudPtr_->at(v,u);
        // if (u>SH_ | v > SW_)
        //     continue;
        // size_t vv = (v + SW_ - pixel_shift_by_row_[u]) % SW_;
        //  keyPointCloudPtr->points
        // keyPointCloudPtr->points[vv] = entirePointCloudPtr_->points[u*SW_ + vv];
        // std::cout << keyPointCloudPtr->points[vv] << std::endl;
    }
   
    
    if(!keyPointCloudPtr->empty())
    {
        publishPointCloud(keyPointCloudPtr);
    }
    
    
    // std::chrono::steady_clock::time_point _end(std::chrono::steady_clock::now());

    // ROS_INFO("Time cost: %f", std::chrono::duration_cast<std::chrono::duration<double>>(_end - _start).count());

}


void lidarImageKeypointOdom::timerCallback(const ros::TimerEvent& event){

    if(img_.empty() | entirePointCloudPtr_->empty()){
        ROS_WARN("Either no image or no point cloud input!");
        return ;
    }
    std::chrono::steady_clock::time_point _start(std::chrono::steady_clock::now());
    float img_gap = std::chrono::duration_cast<std::chrono::duration<double>>(img_time_ - _start).count();
    float pc_gap = std::chrono::duration_cast<std::chrono::duration<double>>(pc_time_ - _start).count();
    if(std::fabs(pc_gap) > 0.2 | std::fabs(img_gap) > 0.2){
        {
        ROS_WARN("No input data!");
        return ;
    }
    }
    cv::Mat mp = img_.clone();

    img_queue_.push(mp);
    if(img_queue_.size() < 2){
        ROS_WARN("Not enough image to match!");
        return;
    }

    cv::Mat img0 = img_queue_.front();
    img_queue_.pop();
    cv::Mat img1 = img_queue_.front();

    cv::Mat resized0, resized1, img_gray0, img_gray1;
    cv::resize(img0, resized0, cv::Size(SW_/2, SH_/2), cv::INTER_AREA);
    cv::resize(img1, resized1, cv::Size(SW_/2, SH_/2), cv::INTER_AREA);
    cv::cvtColor(resized0, img_gray0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(resized1, img_gray1, cv::COLOR_BGR2GRAY);
    // cv::imwrite("/home/xianjia/250-gray.png", img_gray1);

    // if(pts.size() > 0){
    //     cv::Mat img_color = resized1.clone();
    //         for (const auto& kpoint : pts)
    //         {
    //             // const auto& point = kpoint.pt;
    //             cv::circle(img_color, kpoint.pt, 2, cv::Scalar(0, 0, 255), -1);
    //             // points.push_back(point);
    //         }

    //         // cv::imshow("result", resized1);
    //         cv::imwrite("/home/xianjia/250.png", img_color);
    // }
    
    
    if(img_gray0.empty() || img_gray1.empty()){
        return;
    }   
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<cv::KeyPoint> kp0, kp1;
    std::vector<cv::DMatch> matches;
    cv::Mat des0, des1;

    detector->detectAndCompute(img0, cv::Mat(), kp0, des0);
    detector->detectAndCompute(img1, cv::Mat(), kp1, des1);

    matcher->match(des1, des0, matches);


    ROS_INFO("kp0 number: %d, kp1 number: %d,matches number: %d", kp0.size(), kp1.size(), matches.size());

    if(kp0.empty() || kp1.empty() ){
        ROS_WARN("No keypoints detected!");
        return;
    }
   
    std::sort(matches.begin(), matches.end());
    const int match_size = matches.size();

    std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + (int)(match_size * 0.9f));

    pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloudPtr{new pcl::PointCloud<pcl::PointXYZ>};

    // cv::Mat output;
    // std::string output_name = "/home/xianjia/img-" + std::to_string(cnt_)+".jpg";
    // std::cout << output_name << std::endl;
    // cv::drawKeypoints(mp, keypoints,output);
    // cv::imwrite(output_name, output);
    cnt_ = cnt_+ 1;

    keyPointCloudPtr->header = entirePointCloudPtr_->header;
    // keyPointCloudPtr->header.stamp = ros::Time::now();
    pcl_conversions::toPCL(ros::Time::now(), keyPointCloudPtr->header.stamp);
    keyPointCloudPtr->width = entirePointCloudPtr_->width;
    keyPointCloudPtr->height = entirePointCloudPtr_->height;
    keyPointCloudPtr->points.resize(keyPointCloudPtr->width * keyPointCloudPtr->height);
    // keyPointCloudPtr->clear();
    
    // std::cout << keyPointCloudPtr->size() << std::endl;
    for(auto &gm : good_matches){
        size_t u = (size_t) kp1[gm.queryIdx].pt.y * 2;
        size_t v = (size_t) kp1[gm.queryIdx].pt.x * 2;
        if (u>SH_ | v > SW_)
            continue;
        if (u == keyPointCloudPtr->height)
        {
            u--;
        }
        if (v == keyPointCloudPtr->width)
        {
            v--;
        }
        
        // std::cout << u << " " << v << std::endl;
        keyPointCloudPtr->at(v,u) = entirePointCloudPtr_->at(v,u);
        // if (u>SH_ | v > SW_)
        //     continue;
        // size_t vv = (v + SW_ - pixel_shift_by_row_[u]) % SW_;
        //  keyPointCloudPtr->points
        // keyPointCloudPtr->points[vv] = entirePointCloudPtr_->points[u*SW_ + vv];
        // std::cout << keyPointCloudPtr->points[vv] << std::endl;
    }
   
    
    if(!keyPointCloudPtr->empty())
    {
        publishPointCloud(keyPointCloudPtr);
    }
    
    
    std::chrono::steady_clock::time_point _end(std::chrono::steady_clock::now());

    // FIXME: after the subscribing to the keypoint topic, the time cost becomes bigger. 
    ROS_INFO("Time cost: %f", std::chrono::duration_cast<std::chrono::duration<double>>(_end - _start).count());

}

void lidarImageKeypointOdom::publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc){
        sensor_msgs::PointCloud2 rosCloud;
        pcl::toROSMsg(*pc, rosCloud);
        keyPointCloudPub_.publish(rosCloud);
        
        last_received_imu_.header.stamp = ros::Time::now();
        imu_pub_.publish(last_received_imu_);
        ROS_INFO("Point Cloud Published!!!");
}