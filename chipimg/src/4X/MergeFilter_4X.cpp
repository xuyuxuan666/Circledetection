#include "MergeFilter_4X.h"

#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace {

std::unordered_map<int, std::vector<cv::Point2f>>
buildMergedMap4X(const std::vector<Cluster4X>& clusters,
                 const std::vector<GridKeepPoint4X>& keeps)
{
    std::unordered_map<int, std::vector<cv::Point2f>> id2pts;
    id2pts.reserve(clusters.size() * 2 + 16);

    for (const auto& cl : clusters) {
        auto& vec = id2pts[cl.id];
        vec.insert(vec.end(), cl.points.begin(), cl.points.end());
    }

    for (const auto& g : keeps) {
        id2pts[g.cluster_id].push_back(g.pt);
    }
    return id2pts;
}

std::unordered_map<int, cv::Point2f>
buildAnchorMap4X(const std::vector<AnchorInfo4X>& anchors)
{
    std::unordered_map<int, cv::Point2f> id2anchor;
    id2anchor.reserve(anchors.size() + 16);
    for (const auto& a : anchors) {
        id2anchor[a.id] = a.anchor;
    }
    return id2anchor;
}

}

std::vector<MergedClusterPoints4X> mergeAndFilterClusterPoints4X(
    const std::vector<Cluster4X>& clusters,
    const std::vector<GridKeepPoint4X>& keeps,
    const std::vector<AnchorInfo4X>& anchors,
    float up_a, float down_b, float left_c, float right_d)
{

    auto id2pts    = buildMergedMap4X(clusters, keeps);

    auto id2anchor = buildAnchorMap4X(anchors);

    std::vector<MergedClusterPoints4X> out;
    out.reserve(clusters.size());

    for (const auto& cl : clusters) {
        MergedClusterPoints4X mc;
        mc.cluster_id = cl.id;
        mc.row        = cl.row;

        auto it = id2pts.find(cl.id);
        if (it != id2pts.end())
            mc.points = std::move(it->second);

        auto ia = id2anchor.find(cl.id);
        if (ia != id2anchor.end())
            mc.anchor = ia->second;

        if (MF_IsFinitePt4X(mc.anchor) && !mc.points.empty()) {
            const float xmin = mc.anchor.x - left_c;
            const float xmax = mc.anchor.x + right_d;
            const float ymin = mc.anchor.y - up_a;
            const float ymax = mc.anchor.y + down_b;

            std::vector<cv::Point2f> kept;
            kept.reserve(mc.points.size());
            for (const auto& p : mc.points) {
                if (p.x >= xmin && p.x <= xmax && p.y >= ymin && p.y <= ymax) {
                    kept.push_back(p);
                }
            }
            mc.points.swap(kept);
        }

        out.push_back(std::move(mc));
    }

    return out;
}
