#include "MergeFilter_GMY.h"
#include <unordered_map>
#include <algorithm>

using namespace cv;
using namespace std;

namespace {
unordered_map<int, vector<Point2f>>
buildMergedMap_GMY(const vector<ClusterGMY>& clusters,
                   const vector<GridKeepPointGMY>& keeps)
{
    unordered_map<int, vector<Point2f>> id2pts;
    id2pts.reserve(clusters.size()*2+16);
    for (const auto& cl : clusters){
        auto& vec = id2pts[cl.id];
        vec.insert(vec.end(), cl.points.begin(), cl.points.end());
    }
    for (const auto& g : keeps) id2pts[g.cluster_id].push_back(g.pt);
    return id2pts;
}

unordered_map<int, Point2f>
buildAnchorMap_GMY(const vector<AnchorInfoGMY>& anchors)
{
    unordered_map<int, Point2f> id2anchor; id2anchor.reserve(anchors.size()+16);
    for (const auto& a : anchors) id2anchor[a.id] = a.anchor;
    return id2anchor;
}
}

std::vector<MergedClusterPointsGMY> mergeAndFilterClusterPointsGMY(
    const std::vector<ClusterGMY>& clusters,
    const std::vector<GridKeepPointGMY>& keeps,
    const std::vector<AnchorInfoGMY>& anchors,
    float up_a, float down_b, float left_c, float right_d)
{
    auto id2pts    = buildMergedMap_GMY(clusters, keeps);
    auto id2anchor = buildAnchorMap_GMY(anchors);

    vector<MergedClusterPointsGMY> out; out.reserve(clusters.size());
    for (const auto& cl : clusters){
        MergedClusterPointsGMY mc;
        mc.cluster_id = cl.id; mc.row=cl.row;

        auto it = id2pts.find(cl.id);
        if (it != id2pts.end()) mc.points = it->second;

        auto ia = id2anchor.find(cl.id);
        if (ia != id2anchor.end()) mc.anchor = ia->second;

        if (MF_IsFinitePt_GMY(mc.anchor) && !mc.points.empty()){
            const float xmin = mc.anchor.x - left_c;
            const float xmax = mc.anchor.x + right_d;
            const float ymin = mc.anchor.y - up_a;
            const float ymax = mc.anchor.y + down_b;

            vector<Point2f> kept; kept.reserve(mc.points.size());
            for (const auto& p : mc.points){
                if (p.x >= xmin && p.x <= xmax && p.y >= ymin && p.y <= ymax)
                    kept.push_back(p);
            }
            mc.points.swap(kept);
        }
        out.push_back(std::move(mc));
    }
    return out;
}
