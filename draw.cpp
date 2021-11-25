#include <iostream>
#include <string>
#include <opencv2/opencv.hpp> 

using namespace cv;

int main() {
  Mat image = imread("../assets/monalisa-515px.jpg");
  Mat image2(600, 800, CV_8UC3, Scalar(100, 250, 30)); 
  imshow("hello", image);
  imshow("hello", image2);
  waitKey(0);
}