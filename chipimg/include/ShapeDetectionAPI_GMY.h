#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "Cluster_GMY.h"
#include "Anchor_GMY.h"
#include "Grid_GMY.h"
#include "MergeFilter_GMY.h"

struct SD_Position_GMY {
    int x = 0;
    int y = 0;
    int valid = 0;
};

using SD_PositionArray_GMY =
    std::vector<std::vector<std::vector<std::vector<SD_Position_GMY>>>>;

void PerformShapeDetectionGMY(
    const cv::Mat& src16,

    double low_pct = 0.001, double high_pct = 0.010, double gamma_v = 1.4,
    int area_min = 5, float EPS = 35.0f,

    float dy_thresh = 5.0f,
    float dx = 7.0f, float dy = 7.0f, float tol = 4.0f,
    float up_a = 50.0f, float down_b = 5.0f, float left_c = 28.0f, float right_d = 28.0f,

    SD_PositionArray_GMY* out_arr = nullptr
);

void PrintPositionArrayGMY(const SD_PositionArray_GMY& arr);
