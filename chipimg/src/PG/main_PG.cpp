#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

#include "Cluster_PG.h"
#include "Anchor_PG.h"
#include "Grid_PG.h"
#include "MergeFilter_PG.h"
#include "OutputInterface_PG.h"
#include "ShapeDetectionAPI_PG.h"

using namespace std;
using namespace cv;

namespace {
constexpr float kDyThresh  = 7.0f;
constexpr int   kPtRadius  = 2;

constexpr float kA_Up   = 50.0f;
constexpr float kB_Down = 5.0f;
constexpr float kC_Left = 28.0f;
constexpr float kD_Right= 28.0f;

inline bool isFinitePtLocal(const Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

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
    long long low_count  = (long long)llround(total * low_pct);
    long long high_count = (long long)llround(total * (1.0 - high_pct));
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
static void makeEnhancedBaseBGR(const Mat& src16, double low_pct, double high_pct, double gamma_v,
                                Mat& out_bgr) {
    uint16_t a=0, b=65535;
    findPercentile16U(src16, low_pct, high_pct, a, b);
    Mat stretched       = stretch16U(src16, a, b);
    Mat stretched_gamma = gamma16U(stretched, (float)gamma_v);
    Mat eq16;
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8,8));
    clahe->apply(stretched_gamma, eq16);
    Mat eq8; eq16.convertTo(eq8, CV_8U, 1.0/256.0);
    cvtColor(eq8, out_bgr, COLOR_GRAY2BGR);
}
}

int main(int argc, char** argv) {
    string path = (argc >= 2) ? string(argv[1]) : string("../Img/PG/20250611-Q1DAY1-80u.png");
    Mat src16 = imread(path, IMREAD_UNCHANGED);
    if (src16.empty()) { cerr << "读取失败： " << path << "\n"; return -1; }
    if (src16.type() != CV_16UC1) { cerr << "需要 CV_16UC1，当前 type=" << src16.type() << "\n"; return -1; }

    const double low_pct  = 0.013;
    const double high_pct = 0.023;
    const double gamma_v  = 0.86;
    const int   area_min  = 6;
    const float EPS       = 35.0f;

    double   otsu_th = 0.0;
    uint16_t low_v   = 0, high_v = 21793;

    auto clusters = findClustersPG(src16, low_pct, high_pct, gamma_v,
                                   area_min, EPS, &otsu_th, &low_v, &high_v);
    auto anchors = computeAllAnchorsWithFitPG(clusters, kDyThresh);
    const float dx = 10.0f, dy = 19.0f;
    const float tol = 4.0f;
    auto kept = generateAndFilterGridsPG(clusters, anchors, dx, dy, 0.0f);
    auto mergedFiltered = mergeAndFilterClusterPointsPG(
        clusters, kept, anchors, kA_Up, kB_Down, kC_Left, kD_Right
    );

    Mat canvas;
    makeEnhancedBaseBGR(src16, low_pct, high_pct, gamma_v, canvas);

    const Scalar COL_GRID(255, 0, 255);
    const Scalar COL_BOX (100,100,255);

    for (const auto& g : kept) {
        circle(canvas, g.pt, kPtRadius, COL_GRID, FILLED, LINE_AA);
    }
    for (const auto& mc : mergedFiltered) {
        if (!isFinitePtLocal(mc.anchor)) continue;
        float xmin = mc.anchor.x - kC_Left;
        float xmax = mc.anchor.x + kD_Right;
        float ymin = mc.anchor.y - kA_Up;
        float ymax = mc.anchor.y + kB_Down;
        rectangle(canvas, Rect(Point(cvRound(xmin), cvRound(ymin)),
                               Point(cvRound(xmax), cvRound(ymax))),
                  COL_BOX, 1, LINE_AA);
    }

    SD_PositionArray_PG posArr;
    PerformShapeDetectionPG(
        src16,
        low_pct, high_pct, gamma_v,
        area_min, EPS,
        kDyThresh,
        dx, dy, tol,
        kA_Up, kB_Down, kC_Left, kD_Right,
        &posArr
    );

    cout << "\n======= All Well(WR,WC) 3×6 Coordinates =======\n";
    for (int wr = 0; wr < (int)posArr.size(); ++wr) {
        for (int wc = 0; wc < (int)posArr[wr].size(); ++wc) {
            cout << "Well(" << wr << "," << wc << "):\n";
            const auto& plane = posArr[wr][wc];
            for (int i = 0; i < (int)plane.size(); ++i) {
                cout << "   ";
                for (int j = 0; j < (int)plane[i].size(); ++j) {
                    const auto& p = plane[i][j];
                    if (p.valid)
                        cout << "(" << p.x << "," << p.y << ") ";
                    else
                        cout << "[--] ";
                }
                cout << "\n";
            }
        }
    }
    cout << "==============================================\n";

    const string winTitle = "PG | 3x6 Grid + Windows";
    namedWindow(winTitle, WINDOW_AUTOSIZE);
    imshow(winTitle, canvas);
#if CV_VERSION_MAJOR >= 4
    setWindowTitle(winTitle,
                   cv::format("dx=%.1f dy=%.1f | gamma=%.2f | %s",
                              dx, dy, gamma_v, path.c_str()));
#endif
    waitKey(0);
    destroyAllWindows();
    return 0;
}
