#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>

#include "Cluster.h"
#include "Anchor.h"
#include "Grid.h"
#include "MergeFilter.h"
#include "OutputInterface_C5.h"
#include "ShapeDetectionAPI_C5.h"

using namespace std;
using namespace cv;

namespace {
constexpr float kDyThresh  = 7.0f;
constexpr int   kPtRadius  = 2;
constexpr int   kCrossSize = 7;
constexpr int   kCrossThk  = 2;

constexpr float kA_Up    = 48.0f;
constexpr float kB_Down  = 5.0f;
constexpr float kC_Left  = 27.0f;
constexpr float kD_Right = 26.0f;

inline bool isFinitePtLocal(const Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}
inline void drawCross(Mat& img, Point2f p, const Scalar& color,
                      int size = kCrossSize, int thickness = kCrossThk) {
    const Point c(cvRound(p.x), cvRound(p.y));
    line(img, Point(c.x - size, c.y), Point(c.x + size, c.y), color, thickness, LINE_AA);
    line(img, Point(c.x, c.y - size), Point(c.x, c.y + size), color, thickness, LINE_AA);
}
}

int main(int argc, char** argv) {

    string path = (argc >= 2) ? string(argv[1]) : string("../Img/C5/DB20250702-ban3-100u.png");
    Mat src16 = imread(path, IMREAD_UNCHANGED);
    if (src16.empty()) {
        cerr << "读取失败: " << path << "\n";
        return -1;
    }
    if (src16.type() != CV_16UC1) {
        cerr << "图像类型需为 CV_16UC1, 当前为 type=" << src16.type() << "\n";
        return -1;
    }

    const double low_pct  = 0.0041;
    const double high_pct = 0.0379;
    const double gamma_v  = 1.78;
    const int    area_min = 6;
    const float  EPS      = 30.0f;

    const float dx = 9.0f, dy = 9.0f, tol = 3.0f;

    double   otsu_th = 0.0;
    uint16_t low_v = 0, high_v = 65535;

    auto clusters = findClusters(src16, low_pct, high_pct, gamma_v,
                                 area_min, EPS, &otsu_th, &low_v, &high_v);

    auto anchors = computeAllAnchorsWithFit(clusters, kDyThresh);

    auto kept = generateAndFilterGrids(clusters, anchors, dx, dy, tol);

    auto mergedFiltered = mergeAndFilterClusterPoints(
        clusters, kept, anchors,
        kA_Up, kB_Down, kC_Left, kD_Right
    );

    Mat view8; src16.convertTo(view8, CV_8U, 1.0 / 256.0);
    Mat canvas; cvtColor(view8, canvas, COLOR_GRAY2BGR);

    const Scalar COL_BOX    (0, 255, 0);
    const Scalar COL_PTS    (255, 255, 0);
    const Scalar COL_ANCHOR (0, 0, 255);
    const Scalar COL_GRID   (255, 0, 255);
    const Scalar COL_MERGED (0, 255, 255);
    const Scalar COL_WINBOX (100, 100, 255);

    for (size_t i = 0; i < clusters.size(); ++i) {
        const auto& cl = clusters[i];
        const auto& ai = anchors[i];

        rectangle(canvas, cl.bbox, COL_BOX, 1, LINE_AA);

        for (const auto& p : cl.points) {
            circle(canvas, p, kPtRadius, COL_PTS, FILLED, LINE_AA);
        }

        if (isFinitePtLocal(ai.anchor)) {
            drawCross(canvas, ai.anchor, COL_ANCHOR, kCrossSize, kCrossThk);

            const float xmin = ai.anchor.x - kC_Left;
            const float xmax = ai.anchor.x + kD_Right;
            const float ymin = ai.anchor.y - kA_Up;
            const float ymax = ai.anchor.y + kB_Down;
            Rect roi_rect(Point(cvRound(xmin), cvRound(ymin)),
                          Point(cvRound(xmax), cvRound(ymax)));
            rectangle(canvas, roi_rect, COL_WINBOX, 1, LINE_AA);
        }
    }

    for (const auto& g : kept) {
        circle(canvas, g.pt, 2, COL_GRID, FILLED, LINE_AA);
    }

    size_t total_merged_pts = 0;
    for (const auto& mc : mergedFiltered) {
        total_merged_pts += mc.points.size();
        for (const auto& p : mc.points) {
            circle(canvas, p, 2, COL_MERGED, FILLED, LINE_AA);
        }
    }

    auto boxInfos    = ExportClusterBoxesFromSignalsC5(clusters);
    auto circleInfos = ExportCirclesFromMergedC5(mergedFiltered, 3);

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

    cout << "Total kept: " << kept.size() << "\n";
    cout << "Total merged-filtered points: " << total_merged_pts << "\n";

    SD_PositionArray posArr;
    PerformShapeDetectionC5(
        src16,
        low_pct, high_pct, gamma_v,
        area_min, EPS,
        kDyThresh,
        dx, dy, tol,
        kA_Up, kB_Down, kC_Left, kD_Right,
        &posArr
    );
    PrintPositionArrayC5(posArr);

    const string winTitle = "C5 | Clusters + Anchors + Grid + MergedFiltered";
    namedWindow(winTitle, WINDOW_AUTOSIZE);
    imshow(winTitle, canvas);

#if CV_VERSION_MAJOR >= 4
    setWindowTitle(winTitle,
                   cv::format("dx=%.1f dy=%.1f tol=%.1f | a=%.1f b=%.1f c=%.1f d=%.1f | gamma=%.2f low=%u high=%u Otsu=%.1f | %s",
                              dx, dy, tol,
                              kA_Up, kB_Down, kC_Left, kD_Right,
                              gamma_v,
                              static_cast<unsigned>(low_v),
                              static_cast<unsigned>(high_v),
                              otsu_th, path.c_str()));
#endif

    waitKey(0);
    destroyAllWindows();
    return 0;
}
