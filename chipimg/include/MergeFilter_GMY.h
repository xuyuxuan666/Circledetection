#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "Cluster_GMY.h"
#include "Anchor_GMY.h"
#include "Grid_GMY.h"

struct MergedClusterPointsGMY {
    int cluster_id;
    int row;
    cv::Point2f anchor;
    std::vector<cv::Point2f> points;
};

inline bool MF_IsFinitePt_GMY(const cv::Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

std::vector<MergedClusterPointsGMY> mergeAndFilterClusterPointsGMY(
    const std::vector<ClusterGMY>& clusters,
    const std::vector<GridKeepPointGMY>& keeps,
    const std::vector<AnchorInfoGMY>& anchors,
    float up_a, float down_b, float left_c, float right_d);
