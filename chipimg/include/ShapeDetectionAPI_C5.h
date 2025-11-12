#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "Cluster.h"
#include "Anchor.h"
#include "Grid.h"
#include "MergeFilter.h"

struct SD_Position {
    int x = 0;
    int y = 0;
    int valid = 0;
};

using SD_PositionArray =
    std::vector<std::vector<std::vector<std::vector<SD_Position>>>>;

void PerformShapeDetectionC5(
    const cv::Mat& src16,

    double low_pct = 0.0041, double high_pct = 0.0379, double gamma_v = 1.78,
    int area_min = 6, float EPS = 30.0f,

    float dy_thresh = 7.0f,
    float dx = 9.0f, float dy = 9.0f, float tol = 3.0f,
    float up_a = 48.0f, float down_b = 5.0f, float left_c = 27.0f, float right_d = 26.0f,

    SD_PositionArray* out_arr = nullptr
);

void PrintPositionArrayC5(const SD_PositionArray& arr);
