#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "Cluster_4X.h"
#include "Anchor_4X.h"

struct GridKeepPoint4X {
    int cluster_id;
    int row;
    cv::Point2f pt;
};

std::vector<GridKeepPoint4X> generateAndFilterGrids4X(
    const std::vector<Cluster4X>& clusters,
    const std::vector<AnchorInfo4X>& anchors,
    float dx = 10.0f, float dy = 10.0f, float tol = 4.0f);

void drawKeptGridPoints4X(cv::Mat& canvas,
                          const std::vector<GridKeepPoint4X>& keeps,
                          const cv::Scalar& ptColor = cv::Scalar(255, 0, 255),
                          const cv::Scalar& textColor = cv::Scalar(255, 255, 255));
