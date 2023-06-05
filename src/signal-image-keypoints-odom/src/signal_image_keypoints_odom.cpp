#include "signal_image_keypoints_odom/signal_image_keypoints_odom.hpp"

#include </home/xianjia/miniconda3/envs/lidar/include/python3.7m/Python.h>

lidarImageKeypointOdom::lidarImageKeypointOdom(ros::NodeHandle *nh_)
{
    pointCloudSub_ = nh_->subscribe<sensor_msgs::PointCloud2>("/os_cloud_node/points", 10, &lidarImageKeypointOdom::pointCloudCallback, this);

    keyPointsSub_ = nh_->subscribe<sensor_msgs::PointCloud2>("/for_doctor_yu", 10, &lidarImageKeypointOdom::keyPointsCallback, this);

    signalImagesSub_ = nh_->subscribe<sensor_msgs::Image>("/img_node/signal_image", 10, &lidarImageKeypointOdom::signalImageCallback, this);

    keyPointCloudPub_ =  nh_->advertise<sensor_msgs::PointCloud2>("/keypoint_point_cloud", 1);

    timer_ = nh_->createTimer(ros::Duration(0.1), &lidarImageKeypointOdom::timerCallback, this);

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

    ROS_INFO(">>>>>>>> camera pointcloud >>>>>>>>>");
}


void lidarImageKeypointOdom::keyPointsCallback(const sensor_msgs::PointCloud2::ConstPtr& msg){
    pcl::PCLPointCloud2 pcl_pc2;

    pcl_conversions::toPCL(*msg,pcl_pc2);

    pcl::fromPCLPointCloud2(pcl_pc2,*keyPointsPtr_);
}

void lidarImageKeypointOdom::signalImageCallback(const sensor_msgs::ImageConstPtr& msg){ 
    try
    {
       cvPtr_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
       img_ = cvPtr_->image;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    ROS_INFO(">>>>>>>> sigal images received!!! >>>>>>>>>");
}

void lidarImageKeypointOdom::timerCallback(const ros::TimerEvent& event){
    if(img_.empty() | entirePointCloudPtr_->empty()){
        ROS_WARN("Either no image or no point cloud input!");
        return ;
    }
    cv::Mat mp = img_.clone();
    std::chrono::steady_clock::time_point _start(std::chrono::steady_clock::now());
    std::vector<cv::KeyPoint> keypoints;
    if(keyPointsPtr_->empty())
    {
        cv::Mat imgGray;
        cv::cvtColor(mp, imgGray, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
        if(imgGray.empty() || detector == nullptr){
            return;
        }   
        detector->detect(imgGray, keypoints);

        if(keypoints.empty()){
            ROS_WARN("No keypoints detected!");
            return;
        }
        ROS_INFO("keypoint vector size: %d", keypoints.size());
    }
    else
    {
        for(const auto kpc : keyPointsPtr_->points){
            cv::KeyPoint kp;
            kp.pt.x = kpc.x;
            kp.pt.y = kpc.y;
            keypoints.push_back(kp);
        }
    }

	// //Initialize the python instance
	// Py_Initialize();
	
	// //Run a simple string
	// PyRun_SimpleString("from time import time,ctime\n"
	// 					"print('Today is',ctime(time()))\n");

	// //Run a simple file
	// FILE* PScriptFile = fopen("/home/xianjia/temp_ws/src/signal-image-keypoints-odom/script/test.py", "r");
	// if(PScriptFile){
	// 	PyRun_SimpleFile(PScriptFile, "/home/xianjia/temp_ws/src/signal-image-keypoints-odom/script/test.py");
	// 	fclose(PScriptFile);
	// }

	// //Run a python function
	// PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

	// pName = PyUnicode_FromString((char*)"script");
	// pModule = PyImport_Import(pName);
	// pFunc = PyObject_GetAttrString(pModule, (char*)"test");
	// pArgs = PyTuple_Pack(1, PyUnicode_FromString((char*)"Greg"));
	// pValue = PyObject_CallObject(pFunc, pArgs);
	
	// auto result = _PyUnicode_AsString(pValue);
	// std::cout << result << std::endl;

	// //Close the python instance
	// Py_Finalize();


    keyPointCloudPtr_->header = entirePointCloudPtr_->header;
    keyPointCloudPtr_->width = entirePointCloudPtr_->width;
    keyPointCloudPtr_->height = entirePointCloudPtr_->width;
    keyPointCloudPtr_->points.resize(keyPointCloudPtr_->width * keyPointCloudPtr_->height);
    keyPointCloudPtr_->clear();
    
    
    for(auto const kp : keypoints){
        size_t u = (size_t) kp.pt.y;
        size_t v = (size_t) kp.pt.x;
        if (u>SH_ | v > SW_)
            continue;
        size_t vv = (v + SW_ - pixel_shift_by_row_[u]) % SW_;

        keyPointCloudPtr_->points.push_back(entirePointCloudPtr_->points[vv]);
        
        }

    publishPointCloud(keyPointCloudPtr_);
    
    std::chrono::steady_clock::time_point _end(std::chrono::steady_clock::now());

    ROS_INFO("Time cost: %f", std::chrono::duration_cast<std::chrono::duration<double>>(_end - _start).count());

}

void lidarImageKeypointOdom::publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc){
        sensor_msgs::PointCloud2 rosCloud;
        pcl::toROSMsg(*pc, rosCloud);
        keyPointCloudPub_.publish(rosCloud);
        ROS_INFO("Point Cloud Published!!!");
}


