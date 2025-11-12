#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Cluster {
    int id;
    int row;
    std::vector<cv::Rect>    boxes;
    std::vector<cv::Point2f> points;
    cv::Rect                 bbox;
    cv::Point2f              centroid;
};

std::vector<Cluster> findClusters(const cv::Mat& src16,
                                  double low_pct, double high_pct, double gamma_v,
                                  int area_min, float EPS,
                                  double* out_otsu = nullptr,
                                  uint16_t* out_lowv = nullptr,
                                  uint16_t* out_highv = nullptr);
