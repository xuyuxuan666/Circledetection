#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

#include "Cluster.h"
#include "MergeFilter.h"

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
ExportClusterBoxesFromSignalsC5(const std::vector<Cluster>& clusters);

std::vector<POINTPOSITIONINFO_CIRCLE>
ExportCirclesFromMergedC5(const std::vector<MergedClusterPoints>& merged, int radius);
