#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <cmath>
#include "Cluster_4X.h"

struct AnchorInfo4X {
    int id = -1;
    int row = -1;
    cv::Rect bbox;
    cv::Point2f anchor;
    bool has_exact6 = false;
};

bool isFinitePt4X(const cv::Point2f& p);

cv::Point2f computeClusterAnchorTop6_4X(const std::vector<cv::Point2f>& pts,
                                        float dy_thresh = 7.0f,
                                        bool* out_has_exact6 = nullptr);

cv::Point2f linearFitAnchorById4X(const std::vector<std::pair<int, cv::Point2f>>& id_anchor_samples,
                                  int query_id);

std::vector<AnchorInfo4X> computeAllAnchorsWithFit4X(const std::vector<Cluster4X>& clusters,
                                                     float dy_thresh = 7.0f);
