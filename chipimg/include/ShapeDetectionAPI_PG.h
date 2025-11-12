#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "Cluster_PG.h"
#include "Anchor_PG.h"
#include "Grid_PG.h"
#include "MergeFilter_PG.h"

struct SD_Position_PG {
    int x = 0;
    int y = 0;
    int valid = 0;
};

using SD_PositionArray_PG =
    std::vector<std::vector<std::vector<std::vector<SD_Position_PG>>>>;

void PerformShapeDetectionPG(
    const cv::Mat& src16,

    double low_pct = 0.013, double high_pct = 0.023, double gamma_v = 0.86,
    int area_min = 6, float EPS = 35.0f,

    float dy_thresh = 7.0f,
    float dx = 10.0f, float dy = 19.0f, float tol = 4.0f,
    float up_a = 50.0f, float down_b = 5.0f, float left_c = 28.0f, float right_d = 28.0f,

    SD_PositionArray_PG* out_arr = nullptr
);

void PrintPositionArrayPG(const SD_PositionArray_PG& arr);
