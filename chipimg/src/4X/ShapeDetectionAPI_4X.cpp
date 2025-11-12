#include "ShapeDetectionAPI_4X.h"
#include <map>
#include <cmath>
using namespace cv;
using namespace std;

static void genGridRaw4X_local(const Point2f& anchor, float dx, float dy, vector<Point2f>& out) {
    out.clear(); out.reserve(30);
    const float base_x = anchor.x - 23.0f;
    const float base_y = anchor.y - 0.0f;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 6; ++j) {
            out.emplace_back(base_x + j * dx, base_y + i * dy);
        }
    }
}

static void groupClustersByRow(const std::vector<Cluster4X>& clusters,
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
buildClusterPointsMap(const std::vector<MergedClusterPoints4X>& merged)
{
    unordered_map<int, vector<Point2f>> mp; mp.reserve(merged.size()*2+16);
    for (const auto& mc : merged) {
        mp[mc.cluster_id] = mc.points;
    }
    return mp;
}

void PerformShapeDetection(
    const cv::Mat& src16,
    double low_pct, double high_pct, double gamma_v,
    int area_min, float EPS,
    float dy_thresh,
    float dx, float dy, float tol,
    float up_a, float down_b, float left_c, float right_d,
    SD_PositionArray* out_arr)
{
    if (!out_arr) return;
    out_arr->clear();

    if (src16.empty() || src16.type() != CV_16UC1) {

        return;
    }

    double otsu_th = 0.0;
    uint16_t low_v = 0, high_v = 0;

    auto clusters = findClusters4X(src16, low_pct, high_pct, gamma_v,
                                   area_min, EPS, &otsu_th, &low_v, &high_v);
    auto anchors  = computeAllAnchorsWithFit4X(clusters, dy_thresh);
    auto keeps    = generateAndFilterGrids4X(clusters, anchors, dx, dy, tol);
    auto merged   = mergeAndFilterClusterPoints4X(clusters, keeps, anchors,
                                                  up_a, down_b, left_c, right_d);

    vector<vector<int>> rows_idx;
    groupClustersByRow(clusters, rows_idx);
    const int WellRow = (int)rows_idx.size();

    auto id2pts = buildClusterPointsMap(merged);

    out_arr->resize(WellRow);
    for (int wr = 0; wr < WellRow; ++wr) {
        const auto& idxs = rows_idx[wr];
        const int WellCol = (int)idxs.size();

        (*out_arr)[wr].resize(WellCol);
        for (int wc = 0; wc < WellCol; ++wc) {
            int cid = clusters[idxs[wc]].id;

            Point2f anch = anchors[idxs[wc]].anchor;
            vector<Point2f> grid;
            genGridRaw4X_local(anch, dx, dy, grid);

            const auto& detected = id2pts[cid];

            auto& plane = (*out_arr)[wr][wc];
            plane.assign(5, std::vector<SD_Position>(6));

            const float tol2 = tol * tol;
            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 6; ++j) {
                    const Point2f& g = grid[i*6 + j];
                    int best_k = -1; float best_d2 = FLT_MAX;

                    for (int k = 0; k < (int)detected.size(); ++k) {
                        const float dx_ = detected[k].x - g.x;
                        const float dy_ = detected[k].y - g.y;
                        const float d2 = dx_*dx_ + dy_*dy_;
                        if (d2 < best_d2) { best_d2 = d2; best_k = k; }
                    }

                    SD_Position pos;
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

void PrintPositionArray(const SD_PositionArray& arr)
{
    const int WR = (int)arr.size();
    std::cout << "PostionArray WellRow=" << WR << "\n";
    for (int wr = 0; wr < WR; ++wr) {
        const int WC = (int)arr[wr].size();
        std::cout << " Row " << wr << " (WellCol=" << WC << ")\n";
        for (int wc = 0; wc < WC; ++wc) {
            std::cout << "  Well(" << wr << "," << wc << "):\n";
            for (int i = 0; i < 5; ++i) {
                std::cout << "   ";
                for (int j = 0; j < 6; ++j) {
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
