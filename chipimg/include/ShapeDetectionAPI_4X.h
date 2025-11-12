#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "Cluster_4X.h"
#include "Anchor_4X.h"
#include "Grid_4X.h"
#include "MergeFilter_4X.h"

struct SD_Position {
    int x = 0;
    int y = 0;
    int valid = 0;
};

using SD_PositionArray =
    std::vector<std::vector<std::vector<std::vector<SD_Position>>>>;

void PerformShapeDetection(
    const cv::Mat& src16,

    double low_pct = 0.02, double high_pct = 0.0058, double gamma_v = 1.2,
    int area_min = 6, float EPS = 30.0f,

    float dy_thresh = 7.0f,
    float dx = 9.5f, float dy = 9.5f, float tol = 5.0f,
    float up_a = 5.0f, float down_b = 48.0f, float left_c = 28.0f, float right_d = 28.0f,

    SD_PositionArray* out_arr = nullptr
);

void PrintPositionArray(const SD_PositionArray& arr);
