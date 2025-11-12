#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Cluster;
struct AnchorInfo;

struct GridKeepPoint {
    int cluster_id;
    int row;
    cv::Point2f pt;
};

std::vector<GridKeepPoint> generateAndFilterGrids(
    const std::vector<Cluster>& clusters,
    const std::vector<AnchorInfo>& anchors,
    float dx = 10.0f, float dy = 10.0f, float tol = 4.0f);

void drawKeptGridPoints(cv::Mat& canvas,
                        const std::vector<GridKeepPoint>& keeps,
                        const cv::Scalar& ptColor = cv::Scalar(255, 0, 255),
                        const cv::Scalar& textColor = cv::Scalar(255, 255, 255));
