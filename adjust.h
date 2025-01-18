#ifndef ADJUST_H
#define ADJUST_H

#include <vector>
#include <opencv2/opencv.hpp>

void processGroupedCircles(const std::vector<std::vector<cv::Vec3f>>& groupedCircles);
void swapXYInGroupedCircles(std::vector<std::vector<cv::Vec3f>>& groupedCircles);
void flattenGroupedCircles(const std::vector<std::vector<cv::Vec3f>>& groupedCircles,std::vector<cv::Vec3f>& filteredCircles);
void processCircles(std::vector<cv::Vec3f>& filteredCircles);

#endif
