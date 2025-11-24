#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>

#include "Cluster_GMY.h"
#include "Anchor_GMY.h"
#include "Grid_GMY.h"
#include "MergeFilter_GMY.h"
#include "OutputInterface_GMY.h"
#include "ShapeDetectionAPI_GMY.h"

using namespace std;
using namespace cv;

namespace {
constexpr float kDyThresh  = 5.0f;
constexpr int   kPtRadius  = 2;
constexpr int   kCrossSize = 7;
constexpr int   kCrossThk  = 2;

constexpr float kA_Up   = 50.0f;
constexpr float kB_Down = 5.0f;
constexpr float kC_Left = 28.0f;
constexpr float kD_Right= 28.0f;

inline bool isFinitePtLocal(const Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}
inline void drawCross(Mat& img, Point2f p, const Scalar& color,
                      int size = kCrossSize, int thickness = kCrossThk) {
    const Point c(cvRound(p.x), cvRound(p.y));
    line(img, Point(c.x - size, c.y), Point(c.x + size, c.y), color, thickness, LINE_AA);
    line(img, Point(c.x, c.y - size), Point(c.x, c.y + size), color, thickness, LINE_AA);
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
static void makeEnhancedBaseBGR_A(const Mat& src16, double low_pct, double high_pct, double gamma_v,
                                  Mat& out_bgr, uint16_t* used_a=nullptr, uint16_t* used_b=nullptr)
{
    uint16_t a=0, b=65535;
    findPercentile16U(src16, low_pct, high_pct, a, b);
    if (used_a) *used_a = a;
    if (used_b) *used_b = b;

    Mat stretched       = stretch16U(src16, a, b);
    Mat stretched_gamma = gamma16U(stretched, (float)gamma_v);

    Mat eq16;
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8,8));
    clahe->apply(stretched_gamma, eq16);

    Mat eq8; eq16.convertTo(eq8, CV_8U, 1.0/256.0);
    cvtColor(eq8, out_bgr, COLOR_GRAY2BGR);
}

static void placeWindow(const std::string& name, int index, int w, int h, int margin = 40) {
    int col = index % 2;
    int row = index / 2;
    moveWindow(name, col * (w + margin) + margin, row * (h + margin) + margin);
}
}

int main(int argc, char** argv) {
    string path = (argc >= 2) ? string(argv[1]) : string("../Img/GMY60/GMY60-100u.png");
    Mat src16 = imread(path, IMREAD_UNCHANGED);
    if (src16.empty()) {
        cerr << "❌ 读取失败： " << path << "\n";
        return -1;
    }
    if (src16.type() != CV_16UC1) {
        cerr << "❌ 图像类型需为 CV_16UC1，当前为 type=" << src16.type() << "\n";
        return -1;
    }

    const double low_pct  = 0.001;
    const double high_pct = 0.010;
    const double gamma_v  = 1.4;

    const int area_min = 5;
    const float EPS    = 35.0f;

    double otsu_th = 0.0;
    uint16_t low_v = 0, high_v = 0;

    auto clusters = findClustersGMY(src16, low_pct, high_pct, gamma_v,
                                    area_min, EPS, &otsu_th, &low_v, &high_v);

    auto anchors = computeAllAnchorsWithFitGMY(clusters, kDyThresh);

    const float dx = 7.0f, dy = 7.0f, tol = 4.0f;
    auto kept = generateAndFilterGridsGMY(clusters, anchors, dx, dy, tol);

    auto mergedFiltered = mergeAndFilterClusterPointsGMY(
        clusters, kept, anchors, kA_Up, kB_Down, kC_Left, kD_Right
    );

    Mat base;
    uint16_t used_a=0, used_b=0;
    makeEnhancedBaseBGR_A(src16, low_pct, high_pct, gamma_v, base, &used_a, &used_b);

    Mat step1 = base.clone();
    Mat step2 = base.clone();
    Mat step3 = base.clone();
    Mat step4 = base.clone();
    Mat step5 = base.clone();

    const Scalar COL_BOX    (0, 255, 0);
    const Scalar COL_PTS    (255, 255, 0);
    const Scalar COL_ANCHOR (0, 0, 255);
    const Scalar COL_GRID   (255, 0, 255);
    const Scalar COL_MERGED (0, 255, 255);
    const Scalar COL_WINBOX (100, 100, 255);

    for (size_t i = 0; i < clusters.size(); ++i) {
        const auto& cl = clusters[i];
        rectangle(step2, cl.bbox, COL_BOX, 1, LINE_AA);
        for (const auto& p : cl.points) {
            circle(step2, p, kPtRadius, COL_PTS, FILLED, LINE_AA);
        }
    }

    for (size_t i = 0; i < anchors.size(); ++i) {
        const auto& ai = anchors[i];
        if (!isFinitePtLocal(ai.anchor)) continue;
        drawCross(step3, ai.anchor, COL_ANCHOR, kCrossSize, kCrossThk);

        const float xmin = ai.anchor.x - kC_Left;
        const float xmax = ai.anchor.x + kD_Right;
        const float ymin = ai.anchor.y - kA_Up;
        const float ymax = ai.anchor.y + kB_Down;
        Rect roi_rect(Point(cvRound(xmin), cvRound(ymin)),
                      Point(cvRound(xmax), cvRound(ymax)));
        rectangle(step3, roi_rect, COL_WINBOX, 1, LINE_AA);
    }

    for (const auto& g : kept) {
        circle(step4, g.pt, 2, COL_GRID, FILLED, LINE_AA);
    }

    size_t total_mf = 0;
    for (const auto& mc : mergedFiltered) {
        total_mf += mc.points.size();
        for (const auto& p : mc.points) {
            circle(step5, p, 2, COL_MERGED, FILLED, LINE_AA);
        }
    }
    cout << "GMY | Total merged-filtered points: " << total_mf << "\n";

    auto boxInfos    = ExportClusterBoxesFromSignals(clusters);
    auto circleInfos = ExportCirclesFromMerged(mergedFiltered, 3);

    std::cout << "BOX count: " << boxInfos.size() << "\n";
    for (size_t i = 0; i < boxInfos.size(); ++i) {
        const auto& a = boxInfos[i];
        std::cout << i << ") x0=" << a.x0 << ", y0=" << a.y0
                  << ", x1=" << a.x1 << ", y1=" << a.y1 << "\n";
    }
    std::cout << "CIRCLE count: " << circleInfos.size() << "\n";
    for (size_t i = 0; i < circleInfos.size(); ++i) {
        const auto& a = circleInfos[i];
        std::cout << i << ") ix0=" << a.ix0 << ", iy0=" << a.iy0
                  << ", ir=" << a.ir << "\n";
    }

    SD_PositionArray_GMY posArr;
    PerformShapeDetectionGMY(
        src16,
        low_pct, high_pct, gamma_v,
        area_min, EPS,
        kDyThresh,
        dx, dy, tol,
        kA_Up, kB_Down, kC_Left, kD_Right,
        &posArr
    );
    PrintPositionArrayGMY(posArr);

    const string w1 = "GMY Step 1: Enhanced base (Gamma + CLAHE16)";
    const string w2 = "GMY Step 2: + clusters (bbox & raw points)";
    const string w3 = "GMY Step 3: + anchors & asym window";
    const string w4 = "GMY Step 4: + kept grid points";
    const string w5 = "GMY Step 5: + merged & filtered points";

    namedWindow(w1, WINDOW_AUTOSIZE);
    namedWindow(w2, WINDOW_AUTOSIZE);
    namedWindow(w3, WINDOW_AUTOSIZE);
    namedWindow(w4, WINDOW_AUTOSIZE);
    namedWindow(w5, WINDOW_AUTOSIZE);

    imshow(w1, step1);
    imshow(w2, step2);
    imshow(w3, step3);
    imshow(w4, step4);
    imshow(w5, step5);

    int W = step1.cols, H = step1.rows;
    placeWindow(w1, 0, W, H);
    placeWindow(w2, 1, W, H);
    placeWindow(w3, 2, W, H);
    placeWindow(w4, 3, W, H);
    placeWindow(w5, 4, W, H);

#if CV_VERSION_MAJOR >= 4
    setWindowTitle(w1, cv::format("file=%s", path.c_str()));
    setWindowTitle(w2, cv::format("EPS=%.1f | area_min=%d", EPS, area_min));
    setWindowTitle(w3, cv::format("dy_thresh=%.1f | window U=%.1f D=%.1f L=%.1f R=%.1f",
                                  kDyThresh, kA_Up, kB_Down, kC_Left, kD_Right));
    setWindowTitle(w4, cv::format("grid dx=%.1f dy=%.1f tol=%.1f", dx, dy, tol));
    setWindowTitle(w5, cv::format("gamma=%.2f  low=%u  high=%u  Otsu=%.1f",
                                  gamma_v, (unsigned)low_v, (unsigned)high_v, otsu_th));
#endif

    waitKey(0);
    destroyAllWindows();
    return 0;
}
