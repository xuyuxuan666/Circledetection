#include "Cluster.h"
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <cfloat>
#include <algorithm>

using namespace cv;
using namespace std;

static void findPercentile16U(const Mat& img16, double low_pct, double high_pct,
                              uint16_t& low_v, uint16_t& high_v) {
    CV_Assert(img16.type() == CV_16UC1);
    static const int BINS = 65536;
    vector<int> hist(BINS, 0);

    for (int r = 0; r < img16.rows; ++r) {
        const uint16_t* p = img16.ptr<uint16_t>(r);
        for (int c = 0; c < img16.cols; ++c) hist[p[c]]++;
    }
    long long total = 1LL * img16.rows * img16.cols;
    long long low_count  = (long long)std::llround(total * low_pct);
    long long high_count = (long long)std::llround(total * (1.0 - high_pct));

    long long acc = 0; int i = 0;
    for (; i < BINS; ++i) { acc += hist[i]; if (acc >= low_count) break; }
    low_v = (uint16_t)i;

    acc = 0;
    for (i = BINS - 1; i >= 0; --i) { acc += hist[i]; if (acc >= (total - high_count)) break; }
    high_v = (uint16_t)i;

    if (low_v >= high_v) { low_v = 0; high_v = 65535; }
}

static Mat stretch16U(const Mat& src16, uint16_t a, uint16_t b) {
    if (a >= b) return src16.clone();
    Mat f, dst16;
    src16.convertTo(f, CV_32F);
    f = (f - (float)a) * (65535.0f / (float)(b - a));
    threshold(f, f, 65535.0, 65535.0, THRESH_TRUNC);
    threshold(f, f, 0.0, 0.0, THRESH_TOZERO);
    f.convertTo(dst16, CV_16U);
    return dst16;
}

static Mat gamma16U(const Mat& src16, float gamma) {
    Mat f; src16.convertTo(f, CV_32F, 1.0/65535.0);
    pow(f, gamma, f);
    Mat out; f.convertTo(out, CV_16U, 65535.0);
    return out;
}

std::vector<Cluster> findClusters(const cv::Mat& src16,
                                  double low_pct, double high_pct, double gamma_v,
                                  int area_min, float EPS, double* out_otsu,
                                  uint16_t* out_lowv, uint16_t* out_highv) {
    vector<Cluster> clusters;
    if (src16.empty() || src16.type() != CV_16UC1) return clusters;

    uint16_t low_v=0, high_v=65535;
    findPercentile16U(src16, low_pct, high_pct, low_v, high_v);
    Mat stretched = stretch16U(src16, low_v, high_v);
    Mat enhanced  = gamma16U(stretched, (float)gamma_v);

    Mat view8; enhanced.convertTo(view8, CV_8U, 1.0/256.0);
    Mat bin8;
    double otsu_th = threshold(view8, bin8, 0, 255, THRESH_BINARY | THRESH_OTSU);
    if (out_otsu)  *out_otsu  = otsu_th;
    if (out_lowv)  *out_lowv  = low_v;
    if (out_highv) *out_highv = high_v;

    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(bin8, labels, stats, centroids, 8, CV_32S);

    struct Region { Rect bbox; Point2f center; };
    vector<Region> regions;
    regions.reserve(max(0, nLabels-1));
    for (int i = 1; i < nLabels; ++i) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < area_min) continue;
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);
        Point2f c((float)centroids.at<double>(i,0), (float)centroids.at<double>(i,1));
        regions.push_back({Rect(x,y,w,h), c});
    }
    const int N = (int)regions.size();
    if (N == 0) return clusters;

    struct DSU {
        vector<int> p, r;
        DSU(int n): p(n), r(n,0){ iota(p.begin(), p.end(), 0); }
        int find(int x){ return p[x]==x? x : p[x]=find(p[x]); }
        void unite(int a,int b){
            a=find(a); b=find(b);
            if(a==b) return;
            if(r[a]<r[b]) swap(a,b);
            p[b]=a; if(r[a]==r[b]) r[a]++;
        }
    } dsu(N);

    const float EPS2 = EPS * EPS;
    for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
            Point2f d = regions[i].center - regions[j].center;
            if (d.x*d.x + d.y*d.y <= EPS2) dsu.unite(i, j);
        }
    }

    unordered_map<int,int> root2cid;
    vector<int> cid(N, -1);
    int K = 0;
    for (int i = 0; i < N; ++i) {
        int r = dsu.find(i);
        auto it = root2cid.find(r);
        if (it == root2cid.end()) { root2cid[r] = K; cid[i] = K; K++; }
        else cid[i] = it->second;
    }

    clusters.resize(K);
    vector<int> cnt(K, 0);
    for (int k = 0; k < K; ++k) {
        clusters[k].id = k;
        clusters[k].row = -1;
        clusters[k].bbox = Rect();
        clusters[k].centroid = Point2f(0,0);
    }
    for (int i = 0; i < N; ++i) {
        int k = cid[i];
        clusters[k].boxes.push_back(regions[i].bbox);
        clusters[k].points.push_back(regions[i].center);
        clusters[k].centroid += regions[i].center;
        if (cnt[k] == 0) clusters[k].bbox = regions[i].bbox;
        else             clusters[k].bbox |= regions[i].bbox;
        cnt[k]++;
    }
    for (int k = 0; k < K; ++k) {
        if (cnt[k] > 0) clusters[k].centroid *= (1.0f / cnt[k]);
    }

    if (K > 1) {
        const float ROW_EPS = 35.0f;
        const float COL_EPS = 10.0f;

        struct CInfo { int id; Point2f c; };
        vector<CInfo> info; info.reserve(K);
        for (int k = 0; k < K; ++k) info.push_back({k, clusters[k].centroid});

        sort(info.begin(), info.end(), [](const CInfo& a, const CInfo& b){
            if (a.c.y == b.c.y) return a.c.x < b.c.x;
            return a.c.y < b.c.y;
        });

        vector<vector<int>> rows;
        vector<float> row_y_ref;
        for (const auto& ci : info) {
            bool placed = false;
            for (size_t r = 0; r < rows.size(); ++r) {
                if (std::fabs(ci.c.y - row_y_ref[r]) <= ROW_EPS) {
                    rows[r].push_back(ci.id);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                rows.push_back(vector<int>{ci.id});
                row_y_ref.push_back(ci.c.y);
            }
        }

        for (auto& row : rows) {
            sort(row.begin(), row.end(), [&](int a, int b){
                float xa = clusters[a].centroid.x, xb = clusters[b].centroid.x;
                if (std::fabs(xa - xb) > COL_EPS) return xa < xb;
                return clusters[a].centroid.y < clusters[b].centroid.y;
            });
        }

        vector<int> new_order; new_order.reserve(K);
        vector<int> row_of_old(K, -1);
        for (size_t r = 0; r < rows.size(); ++r) {
            for (int old_id : rows[r]) {
                new_order.push_back(old_id);
                row_of_old[old_id] = static_cast<int>(r);
            }
        }

        vector<Cluster> reordered(K);
        for (int new_id = 0; new_id < K; ++new_id) {
            int old_id = new_order[new_id];
            reordered[new_id] = clusters[old_id];
            reordered[new_id].id  = new_id;
            reordered[new_id].row = row_of_old[old_id];
        }
        clusters.swap(reordered);
    } else if (K == 1) {
        clusters[0].row = 0;
    }

    return clusters;
}
