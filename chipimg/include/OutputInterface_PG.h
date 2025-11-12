#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "Cluster_PG.h"
#include "MergeFilter_PG.h"

struct POINTPOSITIONINFO_BOX {
    int x0;
    int y0;
    int x1;
    int y1;
};

struct POINTPOSITIONINFO_CIRCLE {
    int ix0;
    int iy0;
    int ir;
};

std::vector<POINTPOSITIONINFO_BOX>
ExportClusterBoxesFromSignals(const std::vector<ClusterPG>& clusters);

std::vector<POINTPOSITIONINFO_CIRCLE>
ExportCirclesFromMerged(const std::vector<MergedClusterPointsPG>& merged, int radius);
