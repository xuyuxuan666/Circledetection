#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <map>
#include <algorithm>
#include <limits>
#include <cmath>

#include "Anchor_GMY.h"
#include "Cluster_GMY.h"

using namespace cv;
using namespace std;

namespace {
inline Point2f NaNpt() {
    return Point2f(numeric_limits<float>::quiet_NaN(),
                   numeric_limits<float>::quiet_NaN());
}
inline Point2f mean_pt(const vector<Point2f>& g){
    if (g.empty()) return NaNpt();
    double sx=0, sy=0; for (auto&p:g){ sx+=p.x; sy+=p.y; }
    float inv = 1.0f/static_cast<float>(g.size());
    return Point2f(static_cast<float>(sx*inv), static_cast<float>(sy*inv));
}
}

bool isFinitePtGMY(const cv::Point2f& p){
    return std::isfinite(static_cast<double>(p.x)) &&
           std::isfinite(static_cast<double>(p.y));
}

cv::Point2f computeClusterAnchorBottomLR_GMY(const std::vector<cv::Point2f>& pts,
                                             float dy_thresh,
                                             bool* out_has_exact2)
{
    if (out_has_exact2) *out_has_exact2 = false;
    if (pts.empty()) return NaNpt();

    vector<Point2f> v = pts;
    sort(v.begin(), v.end(), [](const Point2f& a, const Point2f& b){
        if (a.y == b.y) return a.x < b.x;
        return a.y < b.y;
    });

    vector<vector<Point2f>> groups;
    vector<Point2f> cur;
    double run_mean = v[0].y;
    int count = 0;
    for (const auto& p : v){
        if (cur.empty()){
            cur.push_back(p); run_mean=p.y; count=1; continue;
        }
        if (std::fabs(p.y - run_mean) <= dy_thresh){
            cur.push_back(p);
            run_mean = (run_mean*count + p.y) / (count+1);
            ++count;
        } else {
            groups.push_back(cur);
            cur.clear();
            cur.push_back(p); run_mean=p.y; count=1;
        }
    }
    if (!cur.empty()) groups.push_back(std::move(cur));
    if (groups.empty()) return NaNpt();

    size_t bottom_idx = 0;
    float  bottom_mean_y = -1e30f;
    for (size_t i = 0; i < groups.size(); ++i){
        Point2f m = mean_pt(groups[i]);
        if (m.y > bottom_mean_y){
            bottom_mean_y = m.y;
            bottom_idx = i;
        }
    }

    const auto& bottom = groups[bottom_idx];
    if (bottom.size() < 2) {
        return NaNpt();
    }

    auto itL = std::min_element(bottom.begin(), bottom.end(),
                                [](const Point2f& a, const Point2f& b){ return a.x < b.x; });
    auto itR = std::max_element(bottom.begin(), bottom.end(),
                                [](const Point2f& a, const Point2f& b){ return a.x < b.x; });

    const Point2f pL = *itL;
    const Point2f pR = *itR;

    Point2f anchor((pL.x + pR.x) * 0.5f, (pL.y + pR.y) * 0.5f);
    if (out_has_exact2) *out_has_exact2 = true;
    return anchor;
}

static bool linfit(const std::vector<std::pair<float,float>>& xy,
                   float xq, float& ypred)
{
    if (xy.size() < 2) return false;
    double Sx=0, Sy=0, Sxx=0, Sxy=0; double n=(double)xy.size();
    for (auto &kv: xy){
        double x=kv.first, y=kv.second;
        Sx += x; Sy += y; Sxx += x*x; Sxy += x*y;
    }
    double den = (n*Sxx - Sx*Sx);
    if (std::fabs(den) < 1e-12) return false;
    double a = (n*Sxy - Sx*Sy) / den;
    double b = (Sy - a*Sx) / n;
    ypred = static_cast<float>(a*xq + b);
    return true;
}

cv::Point2f linearFitAnchorByIdGMY(const std::vector<std::pair<int, cv::Point2f>>& id_anchor_samples,
                                   int query_id)
{
    if (id_anchor_samples.size() < 2) return NaNpt();
    std::vector<std::pair<float,float>> xs, ys;
    xs.reserve(id_anchor_samples.size());
    ys.reserve(id_anchor_samples.size());
    for (const auto& kv : id_anchor_samples){
        const Point2f& p = kv.second;
        if (!isFinitePtGMY(p)) continue;
        xs.emplace_back((float)kv.first, p.x);
        ys.emplace_back((float)kv.first, p.y);
    }
    if (xs.size() < 2 || ys.size() < 2) return NaNpt();

    float px=0, py=0;
    bool okx = linfit(xs, (float)query_id, px);
    bool oky = linfit(ys, (float)query_id, py);
    if (okx && oky) return Point2f(px,py);
    return NaNpt();
}

std::vector<AnchorInfoGMY> computeAllAnchorsWithFitGMY(const std::vector<ClusterGMY>& clusters,
                                                       float dy_thresh)
{
    std::vector<AnchorInfoGMY> infos; infos.reserve(clusters.size());

    for (const auto& cl : clusters){
        AnchorInfoGMY ai;
        ai.id   = cl.id;
        ai.row  = cl.row;
        ai.bbox = cl.bbox;
        bool ok=false;
        ai.anchor = computeClusterAnchorBottomLR_GMY(cl.points, dy_thresh, &ok);
        ai.has_exact6 = ok;
        infos.push_back(std::move(ai));
    }

    std::map<int, std::vector<int>> row_to_idxs;
    for (int i=0; i<(int)infos.size(); ++i)
        row_to_idxs[infos[i].row].push_back(i);

    for (auto &kv : row_to_idxs){
        const auto& idxs = kv.second;
        std::vector<std::pair<int,Point2f>> samples;
        for (int idx : idxs){
            if (isFinitePtGMY(infos[idx].anchor))
                samples.emplace_back(infos[idx].id, infos[idx].anchor);
        }
        for (int idx : idxs){
            auto& ai = infos[idx];
            if (isFinitePtGMY(ai.anchor)) continue;
            Point2f pred = linearFitAnchorByIdGMY(samples, ai.id);
            if (isFinitePtGMY(pred)) {
                ai.anchor = pred;
            } else {
                const Rect& r = ai.bbox;
                ai.anchor = Point2f(r.x + r.width*0.5f, r.y + r.height*0.5f);
            }
        }
    }

    return infos;
}
