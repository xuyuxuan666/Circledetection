#ifndef ADJUST_H
#define ADJUST_H

#include <vector>
#include <opencv2/opencv.hpp>

// 函数声明
void processGroupedCircles(std::vector<std::vector<cv::Vec3f>>& groupedCircles);
void swapXYInGroupedCircles(std::vector<std::vector<cv::Vec3f>>& groupedCircles);
void flattenGroupedCircles(const std::vector<std::vector<cv::Vec3f>>& groupedCircles, 
                            std::vector<cv::Vec3f>& filteredCircles);

#endif // GROUPED_CIRCLES_PROCESSOR_H
