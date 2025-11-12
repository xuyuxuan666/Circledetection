#include "Grid_PG.h"
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

namespace {
static void genGridRaw_PG(const Point2f& anchor, float dx, float dy, vector<Point2f>& out){
    out.clear();
    out.reserve(18);
    const float base_x = anchor.x - 25.0f;
    const float base_y = anchor.y - 38.0f;
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 6; ++j){
            out.emplace_back(base_x + j * dx, base_y + i * dy);
        }
    }
}
}

std::vector<GridKeepPointPG> generateAndFilterGridsPG(
    const std::vector<ClusterPG>& clusters,
    const std::vector<AnchorInfoPG>& anchors,
    float dx, float dy, float )
{
    vector<GridKeepPointPG> keeps;
    if (clusters.size() != anchors.size()) return keeps;

    for (size_t k = 0; k < clusters.size(); ++k){
        const auto& cl = clusters[k];
        const auto& ai = anchors[k];
        if (!isFinitePtPG(ai.anchor)) continue;

        vector<Point2f> grid;
        genGridRaw_PG(ai.anchor, dx, dy, grid);

        for (const auto& gp : grid){
            keeps.push_back(GridKeepPointPG{ cl.id, cl.row, gp });
        }
    }
    return keeps;
}

void drawKeptGridPointsPG(cv::Mat& canvas,
                          const std::vector<GridKeepPointPG>& keeps,
                          const cv::Scalar& ptColor,
                          const cv::Scalar& textColor)
{
    const int radius = 2;
    const int thickness = FILLED;
    for (const auto& g : keeps){
        circle(canvas, g.pt, radius, ptColor, thickness, LINE_AA);
        const string txt = cv::format("(%.1f, %.1f)", g.pt.x, g.pt.y);
        Point org((int)std::round(g.pt.x) + 3, (int)std::round(g.pt.y) - 3);
        putText(canvas, txt, org, FONT_HERSHEY_SIMPLEX, 0.38, textColor, 1, LINE_AA);
    }
}
