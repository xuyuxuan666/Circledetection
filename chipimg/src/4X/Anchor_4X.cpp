#include "Anchor_4X.h"
#include <algorithm>
#include <limits>
#include <map>

using namespace cv;
using namespace std;

namespace {

constexpr float kEps = 1e-3f;

inline Point2f NaNpt() {
    return Point2f(numeric_limits<float>::quiet_NaN(),
                   numeric_limits<float>::quiet_NaN());
}

inline Point2f mean_pt(const vector<Point2f>& g) {
    if (g.empty()) return NaNpt();
    double sx = 0.0, sy = 0.0;
    for (const auto& p : g) { sx += p.x; sy += p.y; }
    const float inv = 1.0f / static_cast<float>(g.size());
    return Point2f(static_cast<float>(sx * inv), static_cast<float>(sy * inv));
}

static bool linfit(const vector<pair<float,float>>& xy, float xq, float& ypred) {
    if (xy.size() < 2) return false;
    double Sx=0, Sy=0, Sxx=0, Sxy=0;
    const double n = static_cast<double>(xy.size());
    for (const auto& kv : xy) {
        const double x = kv.first, y = kv.second;
        Sx  += x;   Sy  += y;
        Sxx += x*x; Sxy += x*y;
    }
    const double den = (n*Sxx - Sx*Sx);
    if (std::fabs(den) < 1e-12) return false;
    const double a = (n*Sxy - Sx*Sy) / den;
    const double b = (Sy - a*Sx) / n;
    ypred = static_cast<float>(a * xq + b);
    return true;
}
}

bool isFinitePt4X(const Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

cv::Point2f computeClusterAnchorTop6_4X(const std::vector<cv::Point2f>& pts,
                                        float dy_thresh,
                                        bool* out_has_exact6)
{
    if (out_has_exact6) *out_has_exact6 = false;
    if (pts.empty()) return NaNpt();

    vector<Point2f> v = pts;
    sort(v.begin(), v.end(), [](const Point2f& a, const Point2f& b){
        if (std::fabs(a.y - b.y) < kEps) return a.x < b.x;
        return a.y < b.y;
    });

    vector<vector<Point2f>> groups; groups.reserve(v.size());
    vector<Point2f> cur; cur.reserve(v.size());
    double run_mean = v[0].y; int count = 0;

    for (const auto& p : v) {
        if (cur.empty()) {
            cur.push_back(p);
            run_mean = p.y;
            count = 1;
            continue;
        }
        if (std::fabs(p.y - run_mean) <= dy_thresh) {
            cur.push_back(p);
            run_mean = (run_mean * count + p.y) / (count + 1);
            ++count;
        } else {
            groups.push_back(cur);
            cur.clear();
            cur.push_back(p);
            run_mean = p.y;
            count = 1;
        }
    }
    if (!cur.empty()) groups.push_back(std::move(cur));
    if (groups.empty()) return NaNpt();

    size_t top_idx = 0;
    float top_mean_y = +1e30f;
    for (size_t i = 0; i < groups.size(); ++i) {
        Point2f m = mean_pt(groups[i]);
        if (m.y < top_mean_y) { top_mean_y = m.y; top_idx = i; }
    }

    const auto& top = groups[top_idx];
    if (top.size() == 6) {
        if (out_has_exact6) *out_has_exact6 = true;
        return mean_pt(top);
    }
    return NaNpt();
}

cv::Point2f linearFitAnchorById4X(const std::vector<std::pair<int, cv::Point2f>>& id_anchor_samples,
                                  int query_id)
{
    if (id_anchor_samples.size() < 2) return NaNpt();

    vector<pair<float,float>> x_samples, y_samples;
    x_samples.reserve(id_anchor_samples.size());
    y_samples.reserve(id_anchor_samples.size());

    for (const auto& kv : id_anchor_samples) {
        const int id = kv.first;
        const Point2f& p = kv.second;
        if (!isFinitePt4X(p)) continue;
        x_samples.emplace_back(static_cast<float>(id), p.x);
        y_samples.emplace_back(static_cast<float>(id), p.y);
    }
    if (x_samples.size() < 2 || y_samples.size() < 2) return NaNpt();

    float px=0, py=0;
    bool okx = linfit(x_samples, static_cast<float>(query_id), px);
    bool oky = linfit(y_samples, static_cast<float>(query_id), py);
    if (okx && oky) return Point2f(px, py);
    return NaNpt();
}

std::vector<AnchorInfo4X> computeAllAnchorsWithFit4X(const std::vector<Cluster4X>& clusters,
                                                     float dy_thresh)
{
    vector<AnchorInfo4X> infos; infos.reserve(clusters.size());

    for (const auto& cl : clusters) {
        AnchorInfo4X ai;
        ai.id = cl.id;
        ai.row = cl.row;
        ai.bbox = cl.bbox;
        ai.anchor = computeClusterAnchorTop6_4X(cl.points, dy_thresh, &ai.has_exact6);
        infos.push_back(std::move(ai));
    }

    map<int, vector<int>> row_to_indices;
    for (int i = 0; i < (int)infos.size(); ++i) {
        row_to_indices[infos[i].row].push_back(i);
    }

    for (const auto& kv : row_to_indices) {
        const auto& idxs = kv.second;

        vector<pair<int, Point2f>> samples;
        for (int idx : idxs) {
            const auto& ai = infos[idx];
            if (isFinitePt4X(ai.anchor)) {
                samples.emplace_back(ai.id, ai.anchor);
            }
        }

        for (int idx : idxs) {
            auto& ai = infos[idx];
            if (isFinitePt4X(ai.anchor)) continue;
            Point2f pred = linearFitAnchorById4X(samples, ai.id);
            if (isFinitePt4X(pred)) {
                ai.anchor = pred;
            } else {

                const Rect& r = ai.bbox;
                ai.anchor = Point2f(r.x + r.width * 0.5f, r.y + r.height * 0.5f);
            }
        }
    }

    return infos;
}
