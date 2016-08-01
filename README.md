# StereoDisparity

Usage:

mkdir build

cd build

cmake ..

make


Test:

./workspace 0 //for two camera

./workspace 1 left.jpg right.jpg

./workspace 2 left.mp4 right.mp4

( Put test_image or test_video into input_file.) 


Note:

Many librarys are needed.

opencv2
libconfig++
boost
pcl
