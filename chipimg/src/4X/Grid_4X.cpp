#include "Grid_4X.h"
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

namespace {

static void genGridRaw4X(const Point2f& anchor, float dx, float dy, vector<Point2f>& out) {
    out.clear();
    out.reserve(30);
    const float base_x = anchor.x - 23.0f;
    const float base_y = anchor.y - 0.0f;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 6; ++j) {
            out.emplace_back(base_x + j * dx, base_y + i * dy);
        }
    }
}
}

std::vector<GridKeepPoint4X> generateAndFilterGrids4X(
    const std::vector<Cluster4X>& clusters,
    const std::vector<AnchorInfo4X>& anchors,
    float dx, float dy, float tol)
{
    std::vector<GridKeepPoint4X> keeps;
    if (clusters.size() != anchors.size()) return keeps;

    const float tol2 = tol * tol;

    for (size_t k = 0; k < clusters.size(); ++k) {
        const Cluster4X& cl = clusters[k];
        const AnchorInfo4X& ai = anchors[k];

        if (!isFinitePt4X(ai.anchor)) continue;

        vector<Point2f> grid;
        genGridRaw4X(ai.anchor, dx, dy, grid);

        for (const auto& gp : grid) {
            bool close_to_signal = false;
            for (const auto& sp : cl.points) {
                const float dx_ = gp.x - sp.x;
                const float dy_ = gp.y - sp.y;
                if (dx_ * dx_ + dy_ * dy_ <= tol2) {
                    close_to_signal = true;
                    break;
                }
            }

            if (!close_to_signal) {
                keeps.push_back(GridKeepPoint4X{ cl.id, cl.row, gp });
            }
        }
    }

    return keeps;
}

void drawKeptGridPoints4X(cv::Mat& canvas,
                          const std::vector<GridKeepPoint4X>& keeps,
                          const cv::Scalar& ptColor,
                          const cv::Scalar& textColor)
{
    const int radius = 2;
    const int thickness = cv::FILLED;
    for (const auto& g : keeps) {

        circle(canvas, g.pt, radius, ptColor, thickness, LINE_AA);

        const std::string txt = cv::format("(%.1f, %.1f)", g.pt.x, g.pt.y);
        Point org((int)std::round(g.pt.x) + 3, (int)std::round(g.pt.y) - 3);
        putText(canvas, txt, org, FONT_HERSHEY_SIMPLEX, 0.38, textColor, 1, LINE_AA);
    }
}
