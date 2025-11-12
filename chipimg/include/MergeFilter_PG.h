#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "Cluster_PG.h"
#include "Anchor_PG.h"
#include "Grid_PG.h"

struct MergedClusterPointsPG {
    int cluster_id = -1;
    int row = -1;
    cv::Point2f anchor;
    std::vector<cv::Point2f> points;
};

inline bool MF_PG_IsFinite(const cv::Point2f& p){
    return std::isfinite(p.x) && std::isfinite(p.y);
}

std::vector<MergedClusterPointsPG> mergeAndFilterClusterPointsPG(
    const std::vector<ClusterPG>& clusters,
    const std::vector<GridKeepPointPG>& keeps,
    const std::vector<AnchorInfoPG>& anchors,
    float up_a, float down_b, float left_c, float right_d);
