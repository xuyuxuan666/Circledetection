#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <algorithm>

using namespace std;
using namespace cv;

namespace {

const double kLowPct    = 0.0046;
const double kHighPct   = 0.0087;
const double kGamma     = 1.33;
const bool   kDoOtsu    = true;
const double kOtsuScale = 1.0;

const int    kMorphKernelW = 3;
const int    kMorphKernelH = 3;
const int    kDilateIters  = 1;
const int    kErodeIters   = 1;

const int    kAreaMin   = 100;
const int    kAreaMax   = 500;
const float  kEPS_L1    = 30.0f;
const float  kEPS_L2    = 100.0f;

const float  kTwoColGapThresh = 40.0f;
const float  kRowGapEps       = 25.0f;
const int    kMaxCols         = 2;
const int    kMaxRows         = 6;

const float  kRowAlignYThresh = 10.0f;
const float  kColAlignXThresh = 1.0f;

const int    kGridRows  = 2;
const int    kGridCols  = 8;
const int    kPtRadius  = 3;

static void findPercentile16U(const Mat& img16, double low_pct, double high_pct,
                              uint16_t& low_v, uint16_t& high_v)
{
    CV_Assert(img16.type() == CV_16UC1);
    static const int BINS = 65536;
    vector<uint32_t> hist(BINS, 0);
    for (int r = 0; r < img16.rows; ++r) {
        const uint16_t* p = img16.ptr<uint16_t>(r);
        for (int c = 0; c < img16.cols; ++c) hist[p[c]]++;
    }
    long long total = 1LL * img16.rows * img16.cols;
    long long target_low  = (long long)llround(total * low_pct);
    long long target_high = (long long)llround(total * high_pct);

    long long acc = 0; int i = 0;
    for (; i < BINS; ++i) { acc += hist[i]; if (acc >= target_low) break; }
    low_v = (uint16_t)i;

    acc = 0;
    for (i = BINS - 1; i >= 0; --i) { acc += hist[i]; if (acc >= target_high) break; }
    high_v = (uint16_t)i;

    if (low_v >= high_v) { low_v = 0; high_v = 65535; }
}

static Mat stretch16U(const Mat& src16, uint16_t a, uint16_t b)
{
    if (a >= b) return src16.clone();
    Mat f, dst16;
    src16.convertTo(f, CV_32F);
    f = (f - (float)a) * (65535.0f / (float)(b - a));
    threshold(f, f, 65535.0, 65535.0, THRESH_TRUNC);
    threshold(f, f, 0.0, 0.0, THRESH_TOZERO);
    f.convertTo(dst16, CV_16U);
    return dst16;
}

static Mat gamma16U(const Mat& src16, float gamma)
{
    Mat f; src16.convertTo(f, CV_32F, 1.0/65535.0);
    pow(f, gamma, f);
    Mat out; f.convertTo(out, CV_16U, 65535.0);
    return out;
}

struct DSU {
    vector<int> p, r;
    explicit DSU(int n): p(n), r(n,0) { iota(p.begin(), p.end(), 0); }
    int find(int x){ return p[x]==x ? x : p[x]=find(p[x]); }
    void unite(int a,int b){
        a=find(a); b=find(b);
        if(a==b) return;
        if(r[a]<r[b]) swap(a,b);
        p[b]=a; if(r[a]==r[b]) r[a]++;
    }
};

static vector<vector<int>> clusterByEpsGroups(const vector<Point2f>& pts, float eps)
{
    vector<vector<int>> groups;
    const int N = (int)pts.size();
    if (N == 0) return groups;

    DSU dsu(N);
    const float eps2 = eps * eps;
    for (int i=0;i<N;++i){
        for (int j=i+1;j<N;++j){
            Point2f d = pts[i] - pts[j];
            if (d.x*d.x + d.y*d.y <= eps2) dsu.unite(i,j);
        }
    }
    unordered_map<int, vector<int>> root2idxs;
    root2idxs.reserve(N*2);
    for (int i=0;i<N;++i) root2idxs[dsu.find(i)].push_back(i);

    groups.reserve(root2idxs.size());
    for (auto &kv : root2idxs){
        groups.push_back(std::move(kv.second));
    }
    return groups;
}

static vector<Point2f> groupsToCenters(const vector<Point2f>& pts,
                                       const vector<vector<int>>& groups)
{
    vector<Point2f> centers;
    centers.reserve(groups.size());
    for (const auto& g : groups){
        double sx=0, sy=0;
        for (int id : g){ sx += pts[id].x; sy += pts[id].y; }
        float inv = 1.0f/static_cast<float>(g.size());
        centers.emplace_back(static_cast<float>(sx*inv), static_cast<float>(sy*inv));
    }
    return centers;
}

struct GridNode { int row, col; Point2f c; };
struct GridNodeEx { int row, col, idx; Point2f c; };

static vector<GridNode> makeGrid2x8(const vector<Point2f>& centers_l2)
{
    vector<GridNode> nodes;
    if (centers_l2.empty()) return nodes;

    vector<Point2f> v = centers_l2;
    sort(v.begin(), v.end(), [](const Point2f& a, const Point2f& b){
        if (a.y == b.y) return a.x < b.x;
        return a.y < b.y;
    });

    int half = (int)v.size()/2;
    vector<Point2f> row0(v.begin(), v.begin()+half);
    vector<Point2f> row1(v.begin()+half, v.end());

    auto mean_y = [](const vector<Point2f>& t)->float{
        if (t.empty()) return std::numeric_limits<float>::quiet_NaN();
        double s=0; for (auto&p:t) s+=p.y; return (float)(s/t.size());
    };
    if (!row0.empty() && !row1.empty()){
        if (mean_y(row0) > mean_y(row1)) std::swap(row0, row1);
    }

    auto sort_by_x = [](vector<Point2f>& t){
        sort(t.begin(), t.end(), [](const Point2f& a, const Point2f& b){
            if (a.x == b.x) return a.y < b.y;
            return a.x < b.x;
        });
    };
    sort_by_x(row0);
    sort_by_x(row1);

    if ((int)row0.size() > kGridCols) row0.resize(kGridCols);
    if ((int)row1.size() > kGridCols) row1.resize(kGridCols);

    nodes.reserve(row0.size()+row1.size());
    for (int c=0;c<(int)row0.size();++c) nodes.push_back({0, c, row0[c]});
    for (int c=0;c<(int)row1.size();++c) nodes.push_back({1, c, row1[c]});
    return nodes;
}

static vector<GridNodeEx> makeGrid2x8WithIndex(const vector<Point2f>& centers_l2)
{
    vector<GridNodeEx> nodes;
    const int N = (int)centers_l2.size();
    if (N == 0) return nodes;

    vector<int> id(N); iota(id.begin(), id.end(), 0);
    sort(id.begin(), id.end(), [&](int a, int b){
        const auto& A = centers_l2[a]; const auto& B = centers_l2[b];
        if (A.y == B.y) return A.x < B.x;
        return A.y < B.y;
    });

    int half = N/2;
    vector<int> row0(id.begin(), id.begin()+half);
    vector<int> row1(id.begin()+half, id.end());

    auto mean_y_idx = [&](const vector<int>& t)->float{
        if (t.empty()) return std::numeric_limits<float>::quiet_NaN();
        double s=0; for (int k : t) s+=centers_l2[k].y; return (float)(s/t.size());
    };
    if (!row0.empty() && !row1.empty()){
        if (mean_y_idx(row0) > mean_y_idx(row1)) std::swap(row0, row1);
    }

    auto sort_by_x_idx = [&](vector<int>& t){
        sort(t.begin(), t.end(), [&](int a, int b){
            const auto& A = centers_l2[a]; const auto& B = centers_l2[b];
            if (A.x == B.x) return A.y < B.y;
            return A.x < B.x;
        });
    };
    sort_by_x_idx(row0);
    sort_by_x_idx(row1);

    if ((int)row0.size() > kGridCols) row0.resize(kGridCols);
    if ((int)row1.size() > kGridCols) row1.resize(kGridCols);

    nodes.reserve(row0.size()+row1.size());
    for (int c=0;c<(int)row0.size();++c) nodes.push_back({0, c, row0[c], centers_l2[row0[c]]});
    for (int c=0;c<(int)row1.size();++c) nodes.push_back({1, c, row1[c], centers_l2[row1[c]]});
    return nodes;
}

struct CellIndex { int col; int row; };

static vector<int> assignColsByX(const vector<Point2f>& pts)
{
    const int N = (int)pts.size();
    vector<int> col(N, 0);
    if (N <= 1) return col;

    vector<int> idx(N); iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b){ return pts[a].x < pts[b].x; });

    float best_gap = -1.f; int best_pos = -1;
    for (int i=0;i<N-1;++i){
        float gap = pts[idx[i+1]].x - pts[idx[i]].x;
        if (gap > best_gap){ best_gap = gap; best_pos = i; }
    }

    bool two_cols = (best_gap > kTwoColGapThresh) && (best_pos >= 0);
    if (!two_cols) return col;

    for (int i=0;i<=best_pos;++i) col[idx[i]] = 0;
    for (int i=best_pos+1;i<N;++i) col[idx[i]] = 1;
    return col;
}

static vector<int> assignRowsByY(const vector<Point2f>& pts, const vector<int>& col)
{
    const int N = (int)pts.size();
    vector<int> row(N, 0);
    if (N == 0) return row;

    for (int c=0;c<kMaxCols;++c){
        vector<int> idc;
        for (int i=0;i<N;++i) if (col[i]==c) idc.push_back(i);
        if (idc.empty()) continue;

        sort(idc.begin(), idc.end(), [&](int a, int b){ return pts[a].y < pts[b].y; });

        int current_row = 0;
        float last_y = pts[idc.front()].y;
        row[idc.front()] = current_row;

        for (size_t k=1;k<idc.size();++k){
            int i = idc[k];
            float y = pts[i].y;
            if ((y - last_y) > kRowGapEps && current_row+1 < kMaxRows){
                current_row++;
            }
            row[i] = current_row;
            last_y = y;
        }

        for (int i : idc){
            if (row[i] >= kMaxRows) row[i] = kMaxRows-1;
        }
    }
    return row;
}

static int pickBottomRightAnchor(const vector<Point2f>& pts,
                                 const vector<int>& col, const vector<int>& row)
{
    int best = -1;
    for (int i=0;i<(int)pts.size();++i){
        if (best < 0){ best = i; continue; }
        if (col[i] > col[best]) { best = i; continue; }
        if (col[i] < col[best]) { continue; }
        if (row[i] > row[best]) { best = i; continue; }
        if (row[i] < row[best]) { continue; }
        if (pts[i].x > pts[best].x) { best = i; continue; }
        if (pts[i].x < pts[best].x) { continue; }
        if (pts[i].y > pts[best].y) { best = i; continue; }
    }
    return best;
}

static bool median_excluding_self(const vector<float>& values, int self_idx, float& out_median)
{
    if ((int)values.size() <= 1) return false;
    vector<float> v; v.reserve(values.size()-1);
    for (int i=0;i<(int)values.size();++i){
        if (i == self_idx) continue;
        float x = values[i];
        if (std::isfinite(x)) v.push_back(x);
    }
    if (v.empty()) return false;
    nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    out_median = v[v.size()/2];
    return true;
}

static bool mean_excluding_self(const vector<float>& values, int self_idx, float& out_mean)
{
    if ((int)values.size() <= 1) return false;
    double sum = 0.0; int cnt = 0;
    for (int i=0;i<(int)values.size();++i){
        if (i == self_idx) continue;
        float x = values[i];
        if (std::isfinite(x)) { sum += x; cnt++; }
    }
    if (cnt == 0) return false;
    out_mean = static_cast<float>(sum / cnt);
    return true;
}

static void fixAbnormalAnchors_RowDetect_ColXMedianOnly(
    vector<Point2f>& anchors_l2,
    const vector<GridNodeEx>& grid_nodes_index,
    float row_tol )
{
    if (grid_nodes_index.empty()) return;

    vector<vector<int>> rows(2);
    for (int i = 0; i < (int)grid_nodes_index.size(); ++i) {
        int idx = grid_nodes_index[i].idx;
        if (idx < 0 || idx >= (int)anchors_l2.size()) continue;
        if (!std::isfinite(anchors_l2[idx].y) || !std::isfinite(anchors_l2[idx].x)) continue;
        int r = grid_nodes_index[i].row;
        if (r >= 0 && r < 2) rows[r].push_back(i);
    }

    vector<vector<int>> cols(kGridCols);
    for (int i = 0; i < (int)grid_nodes_index.size(); ++i) {
        int idx = grid_nodes_index[i].idx;
        if (idx < 0 || idx >= (int)anchors_l2.size()) continue;
        if (!std::isfinite(anchors_l2[idx].x)) continue;
        int c = grid_nodes_index[i].col;
        if (c >= 0 && c < kGridCols) cols[c].push_back(i);
    }

    for (int r = 0; r < 2; ++r) {
        const auto& node_ids_in_row = rows[r];
        if ((int)node_ids_in_row.size() <= 1) continue;

        vector<float> yvals; yvals.reserve(node_ids_in_row.size());
        for (int k = 0; k < (int)node_ids_in_row.size(); ++k) {
            int inode = node_ids_in_row[k];
            int aidx  = grid_nodes_index[inode].idx;
            yvals.push_back(anchors_l2[aidx].y);
        }

        for (int k = 0; k < (int)node_ids_in_row.size(); ++k) {
            int inode = node_ids_in_row[k];
            int aidx  = grid_nodes_index[inode].idx;
            float self_y = anchors_l2[aidx].y;

            float med_y = 0.f;
            if (!median_excluding_self(yvals, k, med_y)) continue;
            if (!std::isfinite(self_y) || !std::isfinite(med_y)) continue;

            float dy = std::fabs(self_y - med_y);
            if (dy <= row_tol) {

                continue;
            }

            int col = grid_nodes_index[inode].col;
            if (col >= 0 && col < kGridCols) {
                const auto& node_ids_in_col = cols[col];
                if ((int)node_ids_in_col.size() > 1) {
                    vector<float> xvals; xvals.reserve(node_ids_in_col.size());
                    int self_pos_in_col = -1;
                    for (int t = 0; t < (int)node_ids_in_col.size(); ++t) {
                        int inode_col = node_ids_in_col[t];
                        int aidx_col  = grid_nodes_index[inode_col].idx;
                        xvals.push_back(anchors_l2[aidx_col].x);
                        if (inode_col == inode) self_pos_in_col = t;
                    }
                    if (self_pos_in_col >= 0) {
                        float med_x = 0.f;
                        if (median_excluding_self(xvals, self_pos_in_col, med_x) && std::isfinite(med_x)) {
                            anchors_l2[aidx].x = med_x;
                        }
                    }
                }
            }

            float mean_y = 0.f;
            if (mean_excluding_self(yvals, k, mean_y) && std::isfinite(mean_y)) {
                anchors_l2[aidx].y = mean_y;
            }
        }
    }
}

}

int main(int argc, char** argv)
{

    string path = (argc >= 2) ? string(argv[1]) : string("../Img/stdChip/sdc_c2_24060602-20uor.png");
    Mat src16 = imread(path, IMREAD_UNCHANGED);
    if (src16.empty()) {
        cerr << "❌ 读取失败：" << path << "\n";
        return -1;
    }
    if (src16.type() != CV_16UC1) {
        cerr << "❌ 需要 CV_16UC1，当前 type=" << src16.type() << "\n";
        return -1;
    }

    uint16_t a=0, b=0;
    findPercentile16U(src16, kLowPct, kHighPct, a, b);
    Mat stretched16 = stretch16U(src16, a, b);

    Mat enhanced16 = gamma16U(stretched16, (float)kGamma);

    Mat view_original, view_stretched, view_enhanced;
    src16.convertTo(view_original,   CV_8U, 1.0/256.0);
    stretched16.convertTo(view_stretched, CV_8U, 1.0/256.0);
    enhanced16.convertTo(view_enhanced,   CV_8U, 1.0/256.0);

    Mat bin8;
    double otsu_th = 0.0;
    if (kDoOtsu) {
        otsu_th = threshold(view_enhanced, bin8, 0, 255, THRESH_BINARY | THRESH_OTSU);
        if (kOtsuScale != 1.0) {
            otsu_th = std::max(0.0, std::min(255.0, otsu_th * kOtsuScale));
            threshold(view_enhanced, bin8, otsu_th, 255, THRESH_BINARY);
        }
    } else {
        threshold(view_enhanced, bin8, 128, 255, THRESH_BINARY);
    }

    if (kDilateIters > 0 || kErodeIters > 0) {
        Mat kernel = getStructuringElement(MORPH_RECT, Size(kMorphKernelW, kMorphKernelH));
        if (kDilateIters > 0) dilate(bin8, bin8, kernel, Point(-1,-1), kDilateIters);
        if (kErodeIters  > 0) erode (bin8, bin8, kernel, Point(-1,-1), kErodeIters);
    }

    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(bin8, labels, stats, centroids, 8, CV_32S);
    const int total_cc = max(0, nLabels - 1);

    struct Region { Point2f c; int area; int label; Rect bbox; };
    vector<Region> kept_regions; kept_regions.reserve(total_cc);
    for (int lbl=1; lbl<nLabels; ++lbl) {
        int area = stats.at<int>(lbl, CC_STAT_AREA);
        if (area < kAreaMin || area > kAreaMax) continue;
        float cx = (float)centroids.at<double>(lbl, 0);
        float cy = (float)centroids.at<double>(lbl, 1);
        Rect bbox(stats.at<int>(lbl, CC_STAT_LEFT),
                  stats.at<int>(lbl, CC_STAT_TOP),
                  stats.at<int>(lbl, CC_STAT_WIDTH),
                  stats.at<int>(lbl, CC_STAT_HEIGHT));
        kept_regions.push_back({ Point2f(cx,cy), area, lbl, bbox });
    }

    vector<Point2f> centers_l1_input;
    centers_l1_input.reserve(kept_regions.size());
    for (auto &r : kept_regions) centers_l1_input.push_back(r.c);

    vector<vector<int>> l1_groups = clusterByEpsGroups(centers_l1_input, kEPS_L1);
    vector<Point2f>     centers_l1_out = groupsToCenters(centers_l1_input, l1_groups);

    vector<vector<int>> l2_groups = clusterByEpsGroups(centers_l1_out, kEPS_L2);
    vector<Point2f>     centers_l2 = groupsToCenters(centers_l1_out, l2_groups);

    vector<Point2f> anchors_l2; anchors_l2.reserve(l2_groups.size());
    vector<int>     l2_cols_found; l2_cols_found.reserve(l2_groups.size());
    vector<int>     l2_rows_max;   l2_rows_max.reserve(l2_groups.size());
    vector<vector<pair<Point2f, CellIndex>>> l2_points_with_idx; l2_points_with_idx.reserve(l2_groups.size());

    for (const auto& g_l2 : l2_groups){
        vector<Point2f> pts;
        for (int id_l1_center : g_l2){
            for (int idx_orig : l1_groups[id_l1_center]){
                pts.push_back(centers_l1_input[idx_orig]);
            }
        }

        if (pts.empty()){
            l2_cols_found.push_back(0);
            l2_rows_max.push_back(0);
            anchors_l2.push_back(Point2f(std::numeric_limits<float>::quiet_NaN(),
                                         std::numeric_limits<float>::quiet_NaN()));
            l2_points_with_idx.push_back({});
            continue;
        }

        vector<int> col = assignColsByX(pts);
        vector<int> row = assignRowsByY(pts, col);

        int max_col = 0, max_row = 0;
        for (size_t i=0;i<pts.size();++i){
            max_col = std::max(max_col, col[i]);
            max_row = std::max(max_row, row[i]);
        }
        l2_cols_found.push_back(max_col + 1);
        l2_rows_max.push_back(max_row + 1);

        int anchor_idx = pickBottomRightAnchor(pts, col, row);
        anchors_l2.push_back(pts[anchor_idx]);

        vector<pair<Point2f, CellIndex>> pack;
        pack.reserve(pts.size());
        for (size_t i=0;i<pts.size();++i) pack.push_back({pts[i], {col[i], row[i]}});
        l2_points_with_idx.push_back(std::move(pack));
    }

    vector<GridNode>   grid_nodes_show   = makeGrid2x8(centers_l2);
    vector<GridNodeEx> grid_nodes_index  = makeGrid2x8WithIndex(centers_l2);

    fixAbnormalAnchors_RowDetect_ColXMedianOnly(anchors_l2, grid_nodes_index, kRowAlignYThresh);

    Mat vis_enh_bgr, vis_bin_bgr;
    cvtColor(view_enhanced, vis_enh_bgr, COLOR_GRAY2BGR);
    cvtColor(bin8,          vis_bin_bgr, COLOR_GRAY2BGR);

    for (const auto& r : kept_regions) {
        rectangle(vis_bin_bgr, r.bbox, Scalar(0,255,0), 1, LINE_AA);
        circle(vis_bin_bgr, r.c, 2, Scalar(0,255,255), FILLED, LINE_AA);
    }

    for (const auto& c : centers_l1_out) {
        circle(vis_enh_bgr, c, kPtRadius,   Scalar(255,255,0), FILLED, LINE_AA);
        circle(vis_bin_bgr, c, kPtRadius,   Scalar(255,255,0), FILLED, LINE_AA);
    }

    for (const auto& c : centers_l2) {
        circle(vis_enh_bgr, c, kPtRadius+1, Scalar(0,0,255),   FILLED, LINE_AA);
        circle(vis_bin_bgr, c, kPtRadius+1, Scalar(0,0,255),   FILLED, LINE_AA);
    }

    for (const auto& pack : l2_points_with_idx){
        for (const auto& pr : pack){
            const Point2f& p = pr.first;
            const CellIndex& ci = pr.second;
            putText(vis_enh_bgr, cv::format("[%d,%d]", ci.col, ci.row),
                    Point(cvRound(p.x)+3, cvRound(p.y)-3),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(180,180,180), 1, LINE_AA);
            putText(vis_bin_bgr, cv::format("[%d,%d]", ci.col, ci.row),
                    Point(cvRound(p.x)+3, cvRound(p.y)-3),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(180,180,180), 1, LINE_AA);
        }
    }

    for (const auto& n : grid_nodes_show) {
        putText(vis_enh_bgr, cv::format("(%d,%d)", n.row, n.col),
                Point(cvRound(n.c.x)+4, cvRound(n.c.y)-4),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255,255,0), 1, LINE_AA);
        putText(vis_bin_bgr, cv::format("(%d,%d)", n.row, n.col),
                Point(cvRound(n.c.x)+4, cvRound(n.c.y)-4),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255,255,0), 1, LINE_AA);
    }

    for (size_t i=0;i<anchors_l2.size();++i){
        const Point2f& a2 = anchors_l2[i];
        if (!std::isfinite(a2.x) || !std::isfinite(a2.y)) continue;
        circle(vis_enh_bgr, a2, kPtRadius+2, Scalar(255,0,255), 2, LINE_AA);
        circle(vis_bin_bgr, a2, kPtRadius+2, Scalar(255,0,255), 2, LINE_AA);
    }

    const string w0 = "Original 16U";
    const string w1 = "Stretched 16U";
    const string w2 = "Enhanced";
    const string w3 = "Binary (Otsu + Morph)";
    const string w4 = "Binary + KeptCC + L1/L2";
    const string w5 = "Enhanced + L1/L2 + Grid(2x8) + Anchors (Row-abnormal -> x=col-median, y=row-mean)";

    namedWindow(w0, WINDOW_AUTOSIZE);
    namedWindow(w1, WINDOW_AUTOSIZE);
    namedWindow(w2, WINDOW_AUTOSIZE);
    namedWindow(w3, WINDOW_AUTOSIZE);
    namedWindow(w4, WINDOW_AUTOSIZE);
    namedWindow(w5, WINDOW_AUTOSIZE);

    imshow(w0, view_original);
    imshow(w1, view_stretched);
    imshow(w2, view_enhanced);
    imshow(w3, bin8);
    imshow(w4, vis_bin_bgr);
    imshow(w5, vis_enh_bgr);

#if CV_VERSION_MAJOR >= 4
    setWindowTitle(w0, cv::format("file=%s", path.c_str()));
    setWindowTitle(w1, cv::format("Stretched [a=%u, b=%u]  (pLow=%.2f%%, pHigh=%.2f%%)",
                                  (unsigned)a, (unsigned)b, kLowPct*100.0, kHighPct*100.0));
    setWindowTitle(w2, cv::format("Enhanced (pLow=%.2f%%, pHigh=%.2f%%, gamma=%.2f)  [a=%u, b=%u]",
                                  kLowPct*100.0, kHighPct*100.0, kGamma, (unsigned)a, (unsigned)b));
    setWindowTitle(w3, cv::format("Otsu=%.1f (scale=%.2f) | Morph: Dilate x%d, Erode x%d, Kernel=%dx%d",
                                  otsu_th, kOtsuScale, kDilateIters, kErodeIters, kMorphKernelW, kMorphKernelH));
    setWindowTitle(w4, cv::format("CC total=%d | kept=%zu (area %d~%d) | L1=%zu (eps=%.1f) | L2=%zu (eps=%.1f)",
                                  total_cc, kept_regions.size(), kAreaMin, kAreaMax,
                                  (size_t)clusterByEpsGroups(centers_l1_input, kEPS_L1).size(), kEPS_L1,
                                  (size_t)clusterByEpsGroups(centers_l1_out, kEPS_L2).size(), kEPS_L2));
    setWindowTitle(w5, cv::format(
        "Grid(2x8): nodes=%zu | L2=%zu | Anchors=%zu | Abnormal by row(y tol=%.1f) -> x=col-median(others), y=row-mean(others); normal unchanged",
        (size_t)grid_nodes_show.size(), centers_l2.size(),
        anchors_l2.size(), kRowAlignYThresh));
#endif

    cout << cv::format("✅ Enhanced (pLow=%.2f%%, pHigh=%.2f%%, gamma=%.2f) [a=%u, b=%u]\n",
                       kLowPct*100.0, kHighPct*100.0, kGamma, (unsigned)a, (unsigned)b);
    cout << cv::format("   Otsu=%.2f | Morph: Dilate x%d, Erode x%d, Kernel=%dx%d\n",
                       otsu_th, kDilateIters, kErodeIters, kMorphKernelW, kMorphKernelH);
    cout << cv::format("   CC total=%d | kept area[%d,%d]=%zu\n",
                       total_cc, kAreaMin, kAreaMax, kept_regions.size());

    for (const auto& node : grid_nodes_index){
        int idx = node.idx;
        if (idx < 0 || idx >= (int)anchors_l2.size()) continue;
        cout << cv::format("   L2 idx=%d -> row=%d col=%d, anchor=(%.1f, %.1f)\n",
                           idx, node.row, node.col, anchors_l2[idx].x, anchors_l2[idx].y);
    }

    waitKey(0);
    destroyAllWindows();
    return 0;
}
