#include "Anchor_PG.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <map>

using namespace cv;
using namespace std;

namespace {
inline Point2f NaNpt(){
    return Point2f(numeric_limits<float>::quiet_NaN(), numeric_limits<float>::quiet_NaN());
}
inline Point2f mean_pt(const vector<Point2f>& v){
    if (v.empty()) return NaNpt();
    double sx=0, sy=0;
    for (const auto& p: v){ sx+=p.x; sy+=p.y; }
    float inv = 1.0f/static_cast<float>(v.size());
    return Point2f(static_cast<float>(sx*inv), static_cast<float>(sy*inv));
}
inline float fallback_center_x(const Rect& r){
    return r.x + r.width * 0.5f;
}
}

bool isFinitePtPG(const Point2f& p){
    return std::isfinite(p.x) && std::isfinite(p.y);
}

cv::Point2f computeClusterAnchorTop6_PG(const std::vector<cv::Point2f>& pts,
                                        float dy_thresh,
                                        bool* out_has_exact6)
{
    if (out_has_exact6) *out_has_exact6 = false;
    if (pts.empty()) return NaNpt();
    vector<Point2f> v = pts;
    sort(v.begin(), v.end(), [](const Point2f& a, const Point2f& b){
        if (a.y == b.y) return a.x < b.x;
        return a.y < b.y;
    });
    vector<vector<Point2f>> groups;
    vector<Point2f> cur; cur.reserve(v.size());
    double run_y = v[0].y; int cnt = 0;
    for (const auto& p : v){
        if (cur.empty()){
            cur.push_back(p); run_y = p.y; cnt = 1;
            continue;
        }
        if (std::fabs(p.y - run_y) <= dy_thresh){
            cur.push_back(p);
            run_y = (run_y*cnt + p.y) / (cnt+1);
            ++cnt;
        }else{
            groups.push_back(cur);
            cur.clear();
            cur.push_back(p); run_y = p.y; cnt = 1;
        }
    }
    if (!cur.empty()) groups.push_back(std::move(cur));
    if (groups.empty()) return NaNpt();
    size_t bottom_i = 0;
    float  bottom_mean_y = -1e30f;
    for (size_t i = 0; i < groups.size(); ++i){
        Point2f m = mean_pt(groups[i]);
        if (m.y > bottom_mean_y){ bottom_mean_y = m.y; bottom_i = i; }
    }
    const auto& bottom = groups[bottom_i];
    if (!bottom.empty()){
        if (out_has_exact6) *out_has_exact6 = true;
        return mean_pt(bottom);
    }
    return NaNpt();
}

static bool linfit(const vector<pair<float,float>>& xy, float xq, float& ypred){
    if (xy.size()<2) return false;
    double Sx=0,Sy=0,Sxx=0,Sxy=0;
    const double n = static_cast<double>(xy.size());
    for (const auto& kv: xy){
        const double x=kv.first, y=kv.second;
        Sx+=x; Sy+=y; Sxx+=x*x; Sxy+=x*y;
    }
    const double den = n*Sxx - Sx*Sx;
    if (std::fabs(den) < 1e-12) return false;
    const double a = (n*Sxy - Sx*Sy)/den;
    const double b = (Sy - a*Sx)/n;
    ypred = static_cast<float>(a*xq + b);
    return true;
}

cv::Point2f linearFitAnchorByIdPG(const std::vector<std::pair<int, cv::Point2f>>& id_anchor_samples,
                                  int query_id)
{
    if (id_anchor_samples.size()<2) return NaNpt();
    vector<pair<float,float>> xs, ys;
    xs.reserve(id_anchor_samples.size());
    ys.reserve(id_anchor_samples.size());
    for (const auto& kv: id_anchor_samples){
        const int id = kv.first;
        const Point2f& p = kv.second;
        if (!isFinitePtPG(p)) continue;
        xs.emplace_back((float)id, p.x);
        ys.emplace_back((float)id, p.y);
    }
    if (xs.size()<2 || ys.size()<2) return NaNpt();
    float px=0, py=0;
    bool okx = linfit(xs, (float)query_id, px);
    bool oky = linfit(ys, (float)query_id, py);
    return (okx && oky) ? Point2f(px,py) : NaNpt();
}

std::vector<AnchorInfoPG> computeAllAnchorsWithFitPG(const std::vector<ClusterPG>& clusters,
                                                     float dy_thresh)
{
    vector<AnchorInfoPG> infos;
    infos.reserve(clusters.size());
    for (const auto& cl : clusters){
        AnchorInfoPG ai;
        ai.id   = cl.id;
        ai.row  = cl.row;
        ai.bbox = cl.bbox;
        ai.anchor = computeClusterAnchorTop6_PG(cl.points, dy_thresh, &ai.has_exact6);
        infos.push_back(std::move(ai));
    }
    std::map<int, std::vector<int>> row2idx;
    for (int i=0;i<(int)infos.size();++i) row2idx[infos[i].row].push_back(i);
    for (auto &kv : row2idx){
        const auto& idxs = kv.second;
        vector<pair<int,Point2f>> samples;
        for (int idx : idxs){
            if (isFinitePtPG(infos[idx].anchor))
                samples.emplace_back(infos[idx].id, infos[idx].anchor);
        }
        for (int idx : idxs){
            auto& ai = infos[idx];
            if (isFinitePtPG(ai.anchor)) continue;
            Point2f pred = linearFitAnchorByIdPG(samples, ai.id);
            if (isFinitePtPG(pred)) {
                ai.anchor = pred;
            } else {
                const Rect& r = ai.bbox;
                ai.anchor = Point2f(r.x + r.width*0.5f, r.y + r.height*0.5f);
            }
        }
    }
    for (auto &kv : row2idx){
        const auto& idxs = kv.second;
        double sum_y = 0.0; int cnt = 0;
        for (int idx : idxs){
            const auto& a = infos[idx].anchor;
            if (isFinitePtPG(a)){ sum_y += a.y; ++cnt; }
            else {
                sum_y += infos[idx].bbox.y + infos[idx].bbox.height*0.5;
                ++cnt;
            }
        }
        if (cnt == 0) continue;
        const float avg_y = static_cast<float>(sum_y / cnt);
        for (int idx : idxs){
            auto& a = infos[idx].anchor;
            if (!isFinitePtPG(a)) a = Point2f(fallback_center_x(infos[idx].bbox), avg_y);
            else                  a.y = avg_y;
        }
    }
    vector<int> col_of_idx(infos.size(), -1);
    int max_col = -1;
    for (auto &kv : row2idx){
        const auto& idxs = kv.second;
        vector<pair<float,int>> order; order.reserve(idxs.size());
        for (int idx : idxs){
            const auto& a = infos[idx].anchor;
            float x = isFinitePtPG(a) ? a.x : fallback_center_x(infos[idx].bbox);
            order.emplace_back(x, idx);
        }
        sort(order.begin(), order.end(), [](const auto& A, const auto& B){
            return A.first < B.first;
        });
        for (int c = 0; c < (int)order.size(); ++c){
            int idx = order[c].second;
            col_of_idx[idx] = c;
            if (c > max_col) max_col = c;
        }
    }
    if (max_col >= 0){
        for (int c = 0; c <= max_col; ++c){
            double sum_x = 0.0; int cnt = 0;
            for (int i = 0; i < (int)infos.size(); ++i){
                if (col_of_idx[i] != c) continue;
                const auto& a = infos[i].anchor;
                if (isFinitePtPG(a)){ sum_x += a.x; ++cnt; }
                else { sum_x += fallback_center_x(infos[i].bbox); ++cnt; }
            }
            if (cnt == 0) continue;
            const float avg_x = static_cast<float>(sum_x / cnt);
            for (int i = 0; i < (int)infos.size(); ++i){
                if (col_of_idx[i] != c) continue;
                auto& a = infos[i].anchor;
                if (!isFinitePtPG(a)) a = Point2f(avg_x, infos[i].bbox.y + infos[i].bbox.height*0.5f);
                else                  a.x = avg_x;
            }
        }
    }
    return infos;
}
