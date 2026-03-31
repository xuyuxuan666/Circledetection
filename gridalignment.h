#ifndef GRIDALIGNMENT_H
#define GRIDALIGNMENT_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

void alignCirclesToGrid(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& alignedCircles);

#endif
