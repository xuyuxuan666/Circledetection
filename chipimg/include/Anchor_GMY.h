#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

struct ClusterGMY;

struct AnchorInfoGMY {
    int id = -1;
    int row = -1;
    cv::Rect bbox;
    cv::Point2f anchor;
    bool has_exact6 = false;
};

bool isFinitePtGMY(const cv::Point2f& p);

cv::Point2f computeClusterAnchorBottomLR_GMY(
    const std::vector<cv::Point2f>& pts,
    float dy_thresh,
    bool* out_has_exact2
);

cv::Point2f linearFitAnchorByIdGMY(
    const std::vector<std::pair<int, cv::Point2f>>& id_anchor_samples,
    int query_id
);

std::vector<AnchorInfoGMY> computeAllAnchorsWithFitGMY(
    const std::vector<ClusterGMY>& clusters,
    float dy_thresh
);
