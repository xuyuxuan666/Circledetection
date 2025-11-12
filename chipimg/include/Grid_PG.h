#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "Cluster_PG.h"
#include "Anchor_PG.h"

struct GridKeepPointPG {
    int cluster_id = -1;
    int row = -1;
    cv::Point2f pt;
};

std::vector<GridKeepPointPG> generateAndFilterGridsPG(
    const std::vector<ClusterPG>& clusters,
    const std::vector<AnchorInfoPG>& anchors,
    float dx = 10.0f, float dy = 10.0f, float tol = 4.0f);

void drawKeptGridPointsPG(cv::Mat& canvas,
                          const std::vector<GridKeepPointPG>& keeps,
                          const cv::Scalar& ptColor = cv::Scalar(255, 0, 255),
                          const cv::Scalar& textColor = cv::Scalar(255, 255, 255));
