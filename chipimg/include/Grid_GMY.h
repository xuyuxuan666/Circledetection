#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "Cluster_GMY.h"
#include "Anchor_GMY.h"

struct GridKeepPointGMY {
    int cluster_id;
    int row;
    cv::Point2f pt;
};

std::vector<GridKeepPointGMY> generateAndFilterGridsGMY(
    const std::vector<ClusterGMY>& clusters,
    const std::vector<AnchorInfoGMY>& anchors,
    float dx = 10.0f, float dy = 10.0f, float tol = 4.0f);

void drawKeptGridPointsGMY(cv::Mat& canvas,
                           const std::vector<GridKeepPointGMY>& keeps,
                           const cv::Scalar& ptColor = cv::Scalar(255, 0, 255),
                           const cv::Scalar& textColor = cv::Scalar(255, 255, 255));
