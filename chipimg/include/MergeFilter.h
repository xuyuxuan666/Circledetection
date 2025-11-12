#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <cmath>

struct Cluster;
struct AnchorInfo;
struct GridKeepPoint;

struct MergedClusterPoints {
    int cluster_id = -1;
    int row = -1;
    cv::Point2f anchor = {NAN, NAN};
    std::vector<cv::Point2f> points;
};

inline bool MF_IsFinitePt(const cv::Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

std::vector<MergedClusterPoints> mergeAndFilterClusterPoints(
    const std::vector<Cluster>& clusters,
    const std::vector<GridKeepPoint>& keeps,
    const std::vector<AnchorInfo>& anchors,
    float up_a   = 48.0f,
    float down_b = 5.0f,
    float left_c = 26.0f,
    float right_d= 26.0f);
