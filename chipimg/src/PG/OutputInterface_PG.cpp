#include "OutputInterface_PG.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>

using namespace cv;
using namespace std;

std::vector<POINTPOSITIONINFO_BOX>
ExportClusterBoxesFromSignals(const std::vector<ClusterPG>& clusters)
{
    std::vector<POINTPOSITIONINFO_BOX> out;
    out.reserve(clusters.size());

    for (const auto& cl : clusters) {
        if (cl.points.empty()) continue;

        float minx = std::numeric_limits<float>::infinity();
        float miny = std::numeric_limits<float>::infinity();
        float maxx = -std::numeric_limits<float>::infinity();
        float maxy = -std::numeric_limits<float>::infinity();

        for (const auto& p : cl.points) {
            if (p.x < minx) minx = p.x;
            if (p.y < miny) miny = p.y;
            if (p.x > maxx) maxx = p.x;
            if (p.y > maxy) maxy = p.y;
        }

        POINTPOSITIONINFO_BOX box{
            cvRound(minx), cvRound(miny),
            cvRound(maxx), cvRound(maxy)
        };
        out.push_back(box);
    }
    return out;
}

std::vector<POINTPOSITIONINFO_CIRCLE>
ExportCirclesFromMerged(const std::vector<MergedClusterPointsPG>& merged, int radius)
{
    std::vector<POINTPOSITIONINFO_CIRCLE> out;
    size_t total = 0;
    for (const auto& mc : merged) total += mc.points.size();
    out.reserve(total);

    for (const auto& mc : merged) {
        for (const auto& p : mc.points) {
            POINTPOSITIONINFO_CIRCLE c{
                cvRound(p.x), cvRound(p.y), radius
            };
            out.push_back(c);
        }
    }
    return out;
}
