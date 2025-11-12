#include "MergeFilter_PG.h"
#include <unordered_map>

using namespace cv;
using namespace std;

namespace {

static unordered_map<int, vector<Point2f>>
buildMerged(const vector<ClusterPG>& clusters, const vector<GridKeepPointPG>& keeps){
    unordered_map<int, vector<Point2f>> mp; mp.reserve(clusters.size()*2+16);
    for (const auto& cl : clusters){
        auto& v = mp[cl.id];
        v.insert(v.end(), cl.points.begin(), cl.points.end());
    }
    for (const auto& g : keeps) mp[g.cluster_id].push_back(g.pt);
    return mp;
}
static unordered_map<int, Point2f>
buildAnchor(const vector<AnchorInfoPG>& anchors){
    unordered_map<int, Point2f> mp; mp.reserve(anchors.size()+16);
    for (const auto& a : anchors) mp[a.id] = a.anchor;
    return mp;
}

}

std::vector<MergedClusterPointsPG> mergeAndFilterClusterPointsPG(
    const std::vector<ClusterPG>& clusters,
    const std::vector<GridKeepPointPG>& keeps,
    const std::vector<AnchorInfoPG>& anchors,
    float up_a, float down_b, float left_c, float right_d)
{
    auto id2pts = buildMerged(clusters, keeps);
    auto id2anc = buildAnchor(anchors);

    vector<MergedClusterPointsPG> out; out.reserve(clusters.size());
    for (const auto& cl : clusters){
        MergedClusterPointsPG mc;
        mc.cluster_id = cl.id;
        mc.row        = cl.row;

        auto it = id2pts.find(cl.id);
        if (it != id2pts.end()) mc.points = std::move(it->second);

        auto ia = id2anc.find(cl.id);
        mc.anchor = (ia==id2anc.end()? Point2f(numeric_limits<float>::quiet_NaN(),
                                               numeric_limits<float>::quiet_NaN())
                                       : ia->second);

        if (MF_PG_IsFinite(mc.anchor) && !mc.points.empty()){
            const float xmin = mc.anchor.x - left_c;
            const float xmax = mc.anchor.x + right_d;
            const float ymin = mc.anchor.y - up_a;
            const float ymax = mc.anchor.y + down_b;
            vector<Point2f> kept; kept.reserve(mc.points.size());
            for (const auto& p : mc.points){
                if (p.x>=xmin && p.x<=xmax && p.y>=ymin && p.y<=ymax) kept.push_back(p);
            }
            mc.points.swap(kept);
        }
        out.push_back(std::move(mc));
    }
    return out;
}
