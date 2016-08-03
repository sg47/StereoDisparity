#include <stdio.h>
#include <iostream>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudastereo.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <libconfig.h++>
#include <cstdlib>
#include <boost/filesystem.hpp>
#include <boost/date_time.hpp>
#include <iomanip>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/cloud_viewer.h>
#include <unistd.h>
using namespace std;
using namespace cv;
using namespace cv::ximgproc;

const char *windowDisparity = "Disparity";
const char *inputPath = "../input_file/";

//VideoCapture cap0(0);
//VideoCapture cap1(1);

string leftFile, rightFile;

/*
boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
  viewer->addCoordinateSystem ( 1.0 );
  viewer->initCameraParameters ();
  return (viewer);
}


bool loadQMatrix(string file, Mat &Q)
{
    bool success = false;
    try
    {
        FileStorage fStorage(file.c_str(), FileStorage::READ);
        fStorage["Q"] >> Q;
        fStorage.release();
        success = true;
    }
    catch(Exception ex)
    {
    }

    return success;
}


void createAndSavePointCloud(Mat &disparity, Mat &leftImage, Mat &Q, string filename)
{
    pcl::PointCloud<pcl::PointXYZRGB> pointCloud;

    // Read out Q Values for faster access
    double Q03 = Q.at<double>(0, 3);
    double Q13 = Q.at<double>(1, 3);
    double Q23 = Q.at<double>(2, 3);
    double Q32 = Q.at<double>(3, 2);
    double Q33 = Q.at<double>(3, 3);

    for (int i = 0; i < disparity.rows; i++)
    {
        for (int j = 0; j < disparity.cols; j++)
        {
            // Create a new point
            pcl::PointXYZRGB point;

            // Read disparity
            float d = disparity.at<float>(i, j);
            if ( d <= 0 ) continue; //Discard bad pixels

            // Read color
            Vec3b colorValue = leftImage.at<Vec3b>(i, j);
            point.r = static_cast<int>(colorValue[2]);
            point.g = static_cast<int>(colorValue[1]);
            point.b = static_cast<int>(colorValue[0]);

            // Transform 2D -> 3D and normalise to point
            double x = Q03 + j;
            double y = Q13 + i;
            double z = Q23;
            double w = (Q32 * d) + Q33;
            point.x = -x / w;
            point.y = -y / w;
            point.z = z / w;

            // Put point into the cloud
            pointCloud.points.push_back (point);
        }
    }

    // Resize PCL and save to file
    pointCloud.width = pointCloud.points.size();
    pointCloud.height = 1;
    pcl::io::savePCDFileASCII(filename, pointCloud);
}
*/



int main(int argc, char *argv[])
{
    VideoCapture cap0(0);
    VideoCapture cap1(1);

    Mat Size( 240, 320, CV_8U);

    Mat frame0, frame1;
    Mat resize0, resize1;
    Mat blur0, blur1;
    Mat mix0, mix1;



/*
    string intrFile, extrFile;
    intrFile = extrFile = inputPath;
    intrFile += "intrinsics.yml";
    extrFile += "extrinsics.yml";

    FileStorage fs( intrFile, FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open intrinsics.yml");
        return -1;
    }
    Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;

    fs.open( extrFile, FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open extrinsics");
        return -1;
    }

    Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;

    Size img_size = frame0.size();
    Rect roi1, roi2;
    Mat Q;

    stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
*/

    if( argv[1][0] == '0')
    {
	if( !cap0.isOpened() || !cap1.isOpened())
	{
	    cout<<"Cannot open cammera"<<endl;
	    return -1;
	}

	cap0>>frame0;
	cap1>>frame1;
    }
    else if( argv[1][0] == '1')
    {
	leftFile += inputPath;
	leftFile += argv[2];

	rightFile += inputPath;
	rightFile += argv[3];	

	frame0 = imread( leftFile, CV_LOAD_IMAGE_COLOR);
	frame1 = imread( rightFile, CV_LOAD_IMAGE_COLOR);
    }
    else
    {
	leftFile += inputPath;
	leftFile += argv[2];

	rightFile += inputPath;
	rightFile += argv[3];	
    }

    int minDisparity = 55;
    int numDisparities = 8;
    int SADWindowSize = 5;
    int Pi1 = 600;
    int Pi2 = 4000;
    int disp12MaxDiff = 10;
    int preFilterCap = 4;
    int uniquenessRatio = 1;
    int speckleWindowSize = 17;
    int speckleRange = 47;
    int lowThreshold = 20;
    bool fullDP = true;
    int a = 5, b = 10, c = 2;
    int lambda = 8000;
    int sigma = 20;

    namedWindow("Control1", CV_WINDOW_AUTOSIZE);
    namedWindow("Control2", CV_WINDOW_AUTOSIZE);
    createTrackbar("lowThreshold", "Control1", &lowThreshold, 300);
    createTrackbar("minDisparity( -64 ~ 0 )", "Control1", &minDisparity, 64);
    createTrackbar("numDisparities( 16 ~ 240 )", "Control1", &numDisparities, 14);
    createTrackbar("SADWindowSize( 1 ~ 31 )", "Control1", &SADWindowSize, 150);
    createTrackbar("P1", "Control1", &Pi1, 10000);
    createTrackbar("P2", "Control1", &Pi2, 10000);
    createTrackbar("disp12MaxDiff", "Control1", &disp12MaxDiff, 100);
    createTrackbar("preFilterCap", "Control1", &preFilterCap, 10);
    createTrackbar("uniquenessRatio", "Control1", &uniquenessRatio, 10);
    createTrackbar("speckleWindowSize", "Control1", &speckleWindowSize, 255);
    createTrackbar("speckleRange", "Control", &speckleRange, 99);
    createTrackbar("a", "Control2", &a, 20);
    createTrackbar("b", "Control2", &b, 100);
    createTrackbar("c", "Control2", &c, 10);
    createTrackbar("lambda", "Control2", &lambda, 10000);
    createTrackbar("sigma( 0.0 ~ 2 )", "Control2", &sigma, 30);

    
    if( argv[1][0] == '0')
    {
	while(1)
	{
	    cap0>>frame0;
	    cap1>>frame1;

	    frame0.convertTo( frame0, CV_8U);
	    frame1.convertTo( frame1, CV_8U);

	    resize( frame0, resize0, Size.size());
	    resize( frame1, resize1, Size.size());
	
	    bilateralFilter( resize0, blur0, a, b, c);
	    bilateralFilter( resize1, blur1, a, b, c);

	    addWeighted( resize0, 1.9, blur0, -1, 0, mix0);
	    addWeighted( resize1, 1.9, blur1, -1, 0, mix1);

	    imshow("left", mix0);
	    imshow("right", mix1);

	    Mat left_disp, right_disp;
	    Mat filtered_disp;
	    Mat imgDisparity8U;

	    Ptr<StereoSGBM> left_matcher = StereoSGBM::create( minDisparity - 64, 16*(numDisparities+1), 2*SADWindowSize + 1, Pi1, Pi2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, fullDP);
	    left_matcher->setMode(StereoSGBM::MODE_HH);
	    Ptr<DisparityWLSFilter> wls_filter = createDisparityWLSFilter(left_matcher);
	    Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

	    left_matcher->compute( mix0, mix1, left_disp);
	    right_matcher->compute( mix1, mix0, right_disp);

	    wls_filter->setLambda(lambda);
	    wls_filter->setSigmaColor(sigma/10);
	    wls_filter->filter(left_disp, resize0, filtered_disp, right_disp);
	
	    double minVal; double maxVal; Mat mean; Mat stddev; int m, s;

	    minMaxLoc( filtered_disp, &minVal, &maxVal);

	    meanStdDev( filtered_disp, mean, stddev);

	    m = mean.at<double>(0,0);
	    s = stddev.at<double>(0,0);
	
	    printf("Min disp: %f Max value: %f mean: %f stddev: %f\n", minVal, maxVal, m, s);

	    Mat thresH;
	    threshold( filtered_disp, thresH, m + 1*s, m + s, THRESH_TRUNC);

	    minMaxLoc( thresH, &minVal, &maxVal);

	    printf("Min disp: %f Max value: %f mean: %f stddev: %f\n", minVal, maxVal, m, s);

	    thresH.convertTo( imgDisparity8U, CV_8U, 255/(maxVal - minVal));

	    Mat cutRefine;
	    cutRefine = imgDisparity8U.colRange( 16*(numDisparities+1), resize1.cols + minDisparity - 64);

	    namedWindow( windowDisparity, WINDOW_NORMAL);
	    imshow( windowDisparity, cutRefine );

	    imwrite("result/disparity.jpg", cutRefine);

	    string dispFile;
	    dispFile += "result/disp.yml";
	    FileStorage data( dispFile, FileStorage::WRITE);
	    data<<"disp"<<cutRefine;
	    data.release();
/*
	    stringstream file;
	    Mat Q(3, 3, CV_64F);
	    loadQMatrix("../input_file/extrinsics.yml", Q);
	    file << "result/_cloud.pcd";
	    createAndSavePointCloud( cutRefine, frame0, Q, file.str());
*/
	    waitKey(1);
        }
    }
	
    else if( argv[1][0] == '1')
    {
	while(1)
	{
	    frame0.convertTo( frame0, CV_8U);
	    frame1.convertTo( frame1, CV_8U);

	    resize( frame0, resize0, Size.size());
	    resize( frame1, resize1, Size.size());
	
	    bilateralFilter( resize0, blur0, a, b, c);
	    bilateralFilter( resize1, blur1, a, b, c);

	    addWeighted( resize0, 1.9, blur0, -1, 0, mix0);
	    addWeighted( resize1, 1.9, blur1, -1, 0, mix1);

	    imshow("left", mix0);
	    imshow("right", mix1);

	    Mat left_disp, right_disp;
	    Mat filtered_disp;
	    Mat imgDisparity8U;

	    Ptr<StereoSGBM> left_matcher = StereoSGBM::create( minDisparity - 64, 16*(numDisparities+1), 2*SADWindowSize + 1, Pi1, Pi2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, fullDP);
	    left_matcher->setMode(StereoSGBM::MODE_HH);
	    Ptr<DisparityWLSFilter> wls_filter = createDisparityWLSFilter(left_matcher);
	    Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

	    left_matcher->compute( mix0, mix1, left_disp);
	    right_matcher->compute( mix1, mix0, right_disp);

	    wls_filter->setLambda(lambda);
	    wls_filter->setSigmaColor(sigma/10);
	    wls_filter->filter(left_disp, resize0, filtered_disp, right_disp);
	
	    double minVal; double maxVal; Mat mean; Mat stddev; int m, s;

	    minMaxLoc( filtered_disp, &minVal, &maxVal);

	    meanStdDev( filtered_disp, mean, stddev);

	    m = mean.at<double>(0,0);
	    s = stddev.at<double>(0,0);
	
	    printf("Min disp: %f Max value: %f mean: %f stddev: %f\n", minVal, maxVal, m, s);

	    Mat thresH;
	    threshold( filtered_disp, thresH, m + 1*s, m + 1*s, THRESH_TRUNC);

	    thresH.convertTo( imgDisparity8U, CV_8U, 255/(maxVal - minVal));

	    Mat cutRefine;
	    cutRefine = imgDisparity8U.colRange( 16*(numDisparities+1), resize1.cols + minDisparity - 64);

	    namedWindow( windowDisparity, WINDOW_NORMAL);
	    imshow( windowDisparity, cutRefine );

	    imwrite("result/disparity.jpg", cutRefine);

	    string dispFile;
	    dispFile += "result/disp.yml";
	    FileStorage data( dispFile, FileStorage::WRITE);
	    data<<"disp"<<cutRefine;
	    data.release();

	    waitKey(1);
        }
    }

    else
    {
	VideoCapture vid0(leftFile);
	VideoCapture vid1(rightFile);

	while(1)
	{
	    vid0>>frame0;
	    vid1>>frame1;

	    frame0.convertTo( frame0, CV_8U);
	    frame1.convertTo( frame1, CV_8U);

	    resize( frame0, resize0, Size.size());
	    resize( frame1, resize1, Size.size());
	
	    bilateralFilter( resize0, blur0, a, b, c);
	    bilateralFilter( resize1, blur1, a, b, c);

	    addWeighted( resize0, 1.9, blur0, -1, 0, mix0);
	    addWeighted( resize1, 1.9, blur1, -1, 0, mix1);

	    imshow("left", mix0);
	    imshow("right", mix1);

	    Mat left_disp, right_disp;
	    Mat filtered_disp;
	    Mat imgDisparity8U;

	    Ptr<StereoSGBM> left_matcher = StereoSGBM::create( minDisparity - 64, 16*(numDisparities+1), 2*SADWindowSize + 1, Pi1, Pi2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, fullDP);
	    left_matcher->setMode(StereoSGBM::MODE_HH);
	    Ptr<DisparityWLSFilter> wls_filter = createDisparityWLSFilter(left_matcher);
	    Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

	    left_matcher->compute( mix0, mix1, left_disp);
	    right_matcher->compute( mix1, mix0, right_disp);

	    wls_filter->setLambda(lambda);
	    wls_filter->setSigmaColor(sigma/10);
	    wls_filter->filter(left_disp, resize0, filtered_disp, right_disp);
	
	    double minVal; double maxVal; Mat mean; Mat stddev; int m, s;

	    minMaxLoc( filtered_disp, &minVal, &maxVal);

	    meanStdDev( filtered_disp, mean, stddev);

	    m = mean.at<double>(0,0);
	    s = stddev.at<double>(0,0);
	
	    printf("Min disp: %f Max value: %f mean: %f stddev: %f\n", minVal, maxVal, m, s);

	    Mat thresH;
	    threshold( filtered_disp, thresH, m + 1*s, m + 1*s, THRESH_TRUNC);

	    minMaxLoc( thresH, &minVal, &maxVal);

	    printf("Min disp: %f Max value: %f mean: %f stddev: %f\n", minVal, maxVal, m, s);

	    thresH.convertTo( imgDisparity8U, CV_8U, 255/(maxVal - minVal));

	    Mat cutRefine;
	    cutRefine = imgDisparity8U.colRange( 16*(numDisparities+1), resize1.cols + minDisparity - 64);

	    namedWindow( windowDisparity, WINDOW_NORMAL);
	    imshow( windowDisparity, cutRefine );

	    imwrite("result/disparity.jpg", cutRefine);

	    string dispFile;
	    dispFile += "result/disp.yml";
	    FileStorage data( dispFile, FileStorage::WRITE);
	    data<<"disp"<<cutRefine;
	    data.release();

	    waitKey(1);
        }
    }    




/*

    cv::Mat recons3D(imgDisparity8U.size(), CV_32F);

    //Reproject image to 3D
    std::cout << "Reprojecting image to 3D..." << std::endl;
    cv::reprojectImageTo3D(imgDisparity8U, recons3D, Q, false, CV_32F);	

	std::cout << "Creating Point Cloud..." << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

    double px, py, pz;
    uchar pr, pg, pb;

    for (int i = 0; i < frame1.rows; i++)
    {
        uchar* rgb_ptr = frame1.ptr<uchar>(i);
#ifdef CUSTOM_REPROJECT
        uchar* disp_ptr = imgDisparity8U.ptr<uchar>(i);
#else
        float* recons_ptr = imgDisparity8U.ptr<float>(i);
#endif
        for (int j = 0; j < frame0.cols; j++)
        {
            //Get 3D coordinates
#ifdef CUSTOM_REPROJECT
            uchar d = disp_ptr[j];
            if (d == 0) continue; //Discard bad pixels
            double pw = -1.0 * static_cast<double>(d)* Q32 + Q33;
            px = static_cast<double>(j)+Q03;
            py = static_cast<double>(i)+Q13;
            pz = Q23;

            px = px / pw;
            py = py / pw;
            pz = pz / pw;

    #else
            px = recons_ptr[3 * j];
            py = recons_ptr[3 * j + 1];
            pz = recons_ptr[3 * j + 2];
    #endif

            //Get RGB info
            pb = rgb_ptr[3 * j];
            pg = rgb_ptr[3 * j + 1];
            pr = rgb_ptr[3 * j + 2];

            //Insert info into point cloud structure
            pcl::PointXYZRGB point;
            point.x = px;
            point.y = py;
            point.z = pz;
            uint32_t rgb = (static_cast<uint32_t>(pr) << 16 |
                static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb)); //NULL
            point.rgb = *reinterpret_cast<float*>(&rgb);
            point_cloud_ptr->points.push_back(point);
        }
    }
    point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
    point_cloud_ptr->height = 20; //1


    //Create visualizer
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    viewer = createVisualizer(point_cloud_ptr);

    //Main loop
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(10); //100
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
*/







/*
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGB>); 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>); 

    for (int rows = 0; rows < imgDisparity8U.rows; ++rows) { 
        for (int cols = 0; cols < imgDisparity8U.cols; ++cols) { 
            cv::Point3f point = imgDisparity8U.at<cv::Point3f>(rows, cols); 


            pcl::PointXYZ pcl_point(point.x, point.y, point.z); // normal PointCloud 
            pcl::PointXYZRGB pcl_point_rgb;
            pcl_point_rgb.x = point.x;    // rgb PointCloud 
            pcl_point_rgb.y = point.y; 
            pcl_point_rgb.z = point.z; 
            cv::Vec3b intensity = frame1.at<cv::Vec3b>(rows,cols); //BGR 
            uint32_t rgb = (static_cast<uint32_t>(intensity[2]) << 16 | static_cast<uint32_t>(intensity[1]) << 8 | static_cast<uint32_t>(intensity[0])); 
            pcl_point_rgb.rgb = *reinterpret_cast<float*>(&rgb);

            cloud_xyz->push_back(pcl_point); 
            cloud_xyzrgb->push_back(pcl_point_rgb); 
           } 
        } 

     std::cout << "saving a pointcloud to out.pcd\n";

    ////Create visualizer
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  viewer = createVisualizer( cloud_xyzrgb );

  ////Main loop
  while ( !viewer->wasStopped())
  {
    viewer->spinOnce(100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

   //pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
   //viewer.showCloud(cloud_xyzrgb);

waitKey(30000);
*/


    return 0;
}

