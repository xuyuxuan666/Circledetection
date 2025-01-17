#ifndef SORTING_H
#define SORTING_H

// #include <opencv2/opencv.hpp>
// #include <vector>

// // 声明使用的命名空间
// using namespace cv;
// using namespace std;


// double computeSlope(const vector<double>& x, const vector<double>& y);


// vector<vector<int>> groupCirclesByAxis(const vector<Vec3f>& circles, char axis, double threshold);
// void adjustCirclePositions(vector<Vec3f>& circles, const vector<vector<int>>& groups, char axis);
// void alignCircles(vector<Vec3f>& circles);
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

using namespace std;

void alignCircles(std::vector<cv::Vec3f>& circles, std::vector<std::vector<cv::Vec3f>>& groupedCircles, int threshold = 30);

#endif 