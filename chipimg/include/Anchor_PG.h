#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "Cluster_PG.h"

struct AnchorInfoPG {
    int id = -1;
    int row = -1;
    cv::Rect bbox;
    cv::Point2f anchor;
    bool has_exact6 = false;
};

bool isFinitePtPG(const cv::Point2f& p);

cv::Point2f computeClusterAnchorTop6_PG(const std::vector<cv::Point2f>& pts,
                                        float dy_thresh = 7.0f,
                                        bool* out_has_exact6 = nullptr);

cv::Point2f linearFitAnchorByIdPG(const std::vector<std::pair<int, cv::Point2f>>& id_anchor_samples,
                                  int query_id);

std::vector<AnchorInfoPG> computeAllAnchorsWithFitPG(const std::vector<ClusterPG>& clusters,
                                                     float dy_thresh = 7.0f);
