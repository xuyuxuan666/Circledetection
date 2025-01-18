#ifndef SORTING_H
#define SORTING_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

using namespace std;

void alignCircles(std::vector<cv::Vec3f>& circles, std::vector<std::vector<cv::Vec3f>>& groupedCircles, int threshold = 30);

#endif 