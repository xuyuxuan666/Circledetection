#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <cmath>

#include "Cluster_4X.h"
#include "Anchor_4X.h"
#include "Grid_4X.h"

struct MergedClusterPoints4X {
    int cluster_id = -1;
    int row = -1;
    cv::Point2f anchor = {NAN, NAN};
    std::vector<cv::Point2f> points;
};

inline bool MF_IsFinitePt4X(const cv::Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

std::vector<MergedClusterPoints4X> mergeAndFilterClusterPoints4X(
    const std::vector<Cluster4X>& clusters,
    const std::vector<GridKeepPoint4X>& keeps,
    const std::vector<AnchorInfo4X>& anchors,
    float up_a   = 50.0f,
    float down_b = 5.0f,
    float left_c = 28.0f,
    float right_d= 28.0f);
