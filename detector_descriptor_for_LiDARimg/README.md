# Directory structure:
- preprocess: Optimal Preprocessing Configuration Searching for LiDAR-Generated Images
- evaluation: Keypoint Detectors and Descriptors for LiDAR-Generated Images 
- dataset: The original rosbag dataset is available at the [University of Turku servers](https://utufi.sharepoint.com/:f:/s/msteams_0ed7e9/Etwsa7m8hxhMk9H3x-K6DfUBgU3x-ZK9vMeD_V0J2mdHwA). We mainly used the signal image from the dataset in this part.

    * image_subscriber.py: a python code to extract images from rosbag, and save them as png images into a local folder. Sometimes this is convenient to try different ways of processing images.
    * signal_image:a local folder that contains signal image

#  Optimal Preprocessing Configuration Searching for LiDAR-Generated Images


Build and run:
open a terminal:
```
mkdir build && cd build
cmake ..
make
./perfermance_comparison 
```

## Setting

1. set loop range for width and height, including maximum, minimum, and step, to find a best resolution that works fo LiDAR-Generated Images.

```CPP
for(int width =512; width <=4096; width += 128)
for(int height = 32; height <= 256; height += 32)

```

2. try different interpolation method.

```CPP
int interpolation_methods[] = {cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::INTER_LANCZOS4, cv::INTER_AREA};

```

3. Set several bool variable to show the image during the process or not.

```CPP
// visualize options setting
bool Vis_Keypoints_window = false; // visualize Keypoints results
bool bVis = false;  // visualize final match results
bool bVis_rubostness = false;  // visualize results of robustness of detector
bool save_sample = false;  // save one sample keypoint results
```

4. image index of your dataset

```CPP
int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)
int imgStartIndex = 0; // first file index to load 
int imgEndIndex = 1145;// last file index to load
```


## Analysis

The csv files by the algorithms are saved inside the folder: csv_output. Then we use the python codes inside the folder: draw_table to analyze the data.





# Keypoint Detectors and Descriptors for LiDAR-Generated Images 
We compared different detector and descriptor for LiDAR-generated image.

Detector:SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT, SURF

Descriptor:BRIEF, FREAK,BRISK, ORB, AKAZE, SIFT,SURF

And a machine learning method: Superpoint


### Run traditional detector and descriptor:


Make sure you have these basic tools.:
- cmake >= 2.8
- make >= 4.1 (Linux, Mac), 3.81 (Windows)
- OpenCV >= 4.1
- gcc/g++ >= 5.4 

In this repo, we will use some non-free function from opencv, such as sift. These functions are inside [open_contrib](https://github.com/opencv/opencv_contrib). You need to install it seperately, if you only have the normal opencv on your ubuntu.


Build and run:
Go to the directory: traditional_method, and open a terminal:
```
mkdir build && cd build
cmake ..
make
./perfermance_comparison 
```
One sample of traditional method:


### Run superpoint method:

Check the [official superpoint](https://github.com/magicleap/SuperPointPretrainedNetwork). Make sure you have all prerequisites to run.

Run:

```sh
./superpoint_evaluation.py ../../Dataset/signal_image/  --W=2048 --H=64 --cuda
```
--W=2048 --H=128, is the size of the image we use 
--cuda: choose use cuda or not


### Evaluation Metrics

* Evaluation 1: number of keypoints detected in this image
* Evaluation 2.1: robustness (rotation)
* Evaluation 2.2: robustness (scaling)
* Evaluation 2.3: robustness (blurred)
* Evaluation 3: Computational Efficiency：ms
* Evaluation 4: Match Ratio
* Evaluation 5: Distinctiveness of detector
* Evaluation 6: Match Score

We run the algorithms to measure and record different evaluation metrics. The metrics are stored in CSV files for subsequent analysis. We mainly analyzed the data by drawing boxplot. For the detailed explanation, please refer to our paper.


### Runing Option

If you want to modify our codes to run your own dataset, here are some options to set.(Use ctrl+f to search the options that you want to adjust in our source codes.)

#### Traditional method 

```cpp

    // visualize and save options setting
    bool Vis_Keypoints_window = false; // visualize Keypoints results
    bool bVis_result = true;  // visualize final match results
    bool save_sample = true;  // save final match results
    bool bVis_rubostness = false;  // visualize rubostness results

    int imgEndIndex = 1144;   // last file index to load//1144
    // output path
    string csv_output_path = "../csv/" + detectorType + "_" + descriptorType + ".csv";

    // input path
    string imgFullFilename = "../../dataset/signal_image/image_" + imgNumber.str() + imgFileType;

    // pre-process
    int newWidth = 1024;
    int newHeight = 64;
    cv::resize(img, imgGray, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    if (save_sample && imgIndex == 3)
    {
        std::string filename = "../demo_image/out_image_" + detector + "_" + descriptor + ".png";
        cv::imwrite(filename, matchImg);
    }

    // --------------optional : limit number of keypoints (helpful for debugging and learning)--------------
    // because when debugging, we can just check very less keypoints to help us understand.
    bool bLimitKpts = false;
```

#### Superpoint_method

```python

# 1.the key line of preprocess
grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR) 

# 2.the name of output_file
output_file = './Superpoint_cuda.csv'
      
# 3. image_number to run
while image_number < 1145: 

# 4. option of saving one sample image
if image_number == 3:
    filename = './superpoint_sample.png'
    cv2.imwrite(filename, out2)
```

