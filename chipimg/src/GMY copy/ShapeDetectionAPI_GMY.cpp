#include "ShapeDetectionAPI_GMY.h"
#include <map>
#include <cfloat>
#include <cmath>
using namespace cv;
using namespace std;

static void genGridRaw_GMY_local(const Point2f& anchor, float dx, float dy, vector<Point2f>& out) {
    out.clear(); out.reserve(64);
    const float base_x = anchor.x - 25.0f;
    const float base_y = anchor.y - 49.0f;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            out.emplace_back(base_x + j * dx, base_y + i * dy);
        }
    }
}

static void groupClustersByRowGMY(const std::vector<ClusterGMY>& clusters,
                                  std::vector<std::vector<int>>& rows_idx)
{
    std::map<int, std::vector<int>> row2idx;
    for (int i = 0; i < (int)clusters.size(); ++i) {
        row2idx[clusters[i].row].push_back(i);
    }
    rows_idx.clear(); rows_idx.reserve(row2idx.size());
    for (auto& kv : row2idx) {
        auto& vec = kv.second;
        std::sort(vec.begin(), vec.end(), [&](int a, int b){
            return clusters[a].centroid.x < clusters[b].centroid.x;
        });
        rows_idx.push_back(vec);
    }
}

static unordered_map<int, vector<Point2f>>
buildClusterPointsMapGMY(const std::vector<MergedClusterPointsGMY>& merged)
{
    unordered_map<int, vector<Point2f>> mp; mp.reserve(merged.size()*2+16);
    for (const auto& mc : merged) {
        mp[mc.cluster_id] = mc.points;
    }
    return mp;
}

void PerformShapeDetectionGMY(
    const cv::Mat& src16,
    double low_pct, double high_pct, double gamma_v,
    int area_min, float EPS,
    float dy_thresh,
    float dx, float dy, float tol,
    float up_a, float down_b, float left_c, float right_d,
    SD_PositionArray_GMY* out_arr)
{
    if (!out_arr) return;
    out_arr->clear();

    if (src16.empty() || src16.type() != CV_16UC1) {

        return;
    }

    double otsu_th = 0.0;
    uint16_t low_v = 0, high_v = 0;

    auto clusters = findClustersGMY(src16, low_pct, high_pct, gamma_v,
                                    area_min, EPS, &otsu_th, &low_v, &high_v);
    auto anchors  = computeAllAnchorsWithFitGMY(clusters, dy_thresh);
    auto keeps    = generateAndFilterGridsGMY(clusters, anchors, dx, dy, tol);
    auto merged   = mergeAndFilterClusterPointsGMY(clusters, keeps, anchors,
                                                   up_a, down_b, left_c, right_d);

    vector<vector<int>> rows_idx;
    groupClustersByRowGMY(clusters, rows_idx);
    const int WellRow = (int)rows_idx.size();

    auto id2pts = buildClusterPointsMapGMY(merged);

    out_arr->resize(WellRow);
    for (int wr = 0; wr < WellRow; ++wr) {
        const auto& idxs = rows_idx[wr];
        const int WellCol = (int)idxs.size();

        (*out_arr)[wr].resize(WellCol);
        for (int wc = 0; wc < WellCol; ++wc) {
            int cid = clusters[idxs[wc]].id;

            Point2f anch = anchors[idxs[wc]].anchor;
            vector<Point2f> grid;
            genGridRaw_GMY_local(anch, dx, dy, grid);

            const auto it = id2pts.find(cid);
            const vector<Point2f>& detected = (it == id2pts.end())
                                              ? * (new vector<Point2f>())
                                              : it->second;

            auto& plane = (*out_arr)[wr][wc];
            plane.assign(8, std::vector<SD_Position_GMY>(8));

            const float tol2 = tol * tol;
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    const Point2f& g = grid[i*8 + j];
                    int best_k = -1; float best_d2 = FLT_MAX;

                    for (int k = 0; k < (int)detected.size(); ++k) {
                        const float dx_ = detected[k].x - g.x;
                        const float dy_ = detected[k].y - g.y;
                        const float d2 = dx_*dx_ + dy_*dy_;
                        if (d2 < best_d2) { best_d2 = d2; best_k = k; }
                    }

                    SD_Position_GMY pos;
                    if (best_k >= 0 && best_d2 <= tol2) {
                        pos.x = cvRound(detected[best_k].x);
                        pos.y = cvRound(detected[best_k].y);
                        pos.valid = 1;
                    } else {
                        pos.x = cvRound(g.x);
                        pos.y = cvRound(g.y);
                        pos.valid = 0;
                    }
                    plane[i][j] = pos;
                }
            }
        }
    }
}

void PrintPositionArrayGMY(const SD_PositionArray_GMY& arr)
{
    const int WR = (int)arr.size();
    std::cout << "PostionArray (GMY) WellRow=" << WR << "\n";
    for (int wr = 0; wr < WR; ++wr) {
        const int WC = (int)arr[wr].size();
        std::cout << " Row " << wr << " (WellCol=" << WC << ")\n";
        for (int wc = 0; wc < WC; ++wc) {
            std::cout << "  Well(" << wr << "," << wc << "):\n";
            for (int i = 0; i < 8; ++i) {
                std::cout << "   ";
                for (int j = 0; j < 8; ++j) {
                    const auto& p = arr[wr][wc][i][j];
                    if (p.valid)
                        std::cout << "(" << p.x << "," << p.y << ") ";
                    else
                        std::cout << "[--] ";
                }
                std::cout << "\n";
            }
        }
    }
}
