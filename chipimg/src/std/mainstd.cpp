#include "OutputInterface_std.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <cmath>

using namespace std;
using namespace cv;

static void draw_points(
    Mat& img,
    _POINTPOSITIONINFO (&pos)[WellRow][WellCol][PointRow][PointCol])
{
    for (int wr = 0; wr < WellRow; ++wr)
    for (int wc = 0; wc < WellCol; ++wc)
    for (int pr = 0; pr < PointRow; ++pr)
    for (int pc = 0; pc < PointCol; ++pc) {
        const auto& p = pos[wr][wc][pr][pc];
        if (!p.valid || !isfinite(p.x) || !isfinite(p.y)) continue;
        Point2f pt(p.x, p.y);
        Scalar color = p.measured ? Scalar(0,255,0) : Scalar(0,0,255); // M=绿, F=红
        circle(img, pt, 4, color, FILLED, LINE_AA);
        char label[32];
        snprintf(label, sizeof(label), "(%d,%d)-(%d,%d)", wr, wc, pr, pc);
        putText(img, label, pt + Point2f(5, -5),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255,255,0), 1, LINE_AA);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <16bit_gray_image>\n";
        cerr << "Note : image must be CV_16UC1 (16-bit single-channel).\n";
        return 1;
    }

    string path = argv[1];
    Mat src16 = imread(path, IMREAD_UNCHANGED);
    if (src16.empty()) { cerr << "❌ Cannot read image: " << path << "\n"; return 2; }
    if (src16.type() != CV_16UC1) { cerr << "❌ Must be CV_16UC1.\n"; return 3; }

    cout << "✅ Image loaded: " << path << "  size=" << src16.cols << "x" << src16.rows << "\n";

    _POINTPOSITIONINFO pos[WellRow][WellCol][PointRow][PointCol];
    PerformShapeDetectionDyn(src16.ptr<ushort>(), src16.cols, src16.rows, pos);

    for (int wr = 0; wr < WellRow; ++wr) {
        for (int wc = 0; wc < WellCol; ++wc) {
            cout << "Well(" << wr << "," << wc << "):\n";
            for (int r = 0; r < PointRow; ++r) {
                for (int c = 0; c < PointCol; ++c) {
                    const auto& p = pos[wr][wc][r][c];
                    if (!p.valid) continue;
                    cout << "  P(" << r << "," << c << "): ("
                         << p.x << ", " << p.y << ") "
                         << (p.measured ? "[M]" : "[F]") << "\n";
                }
            }
        }
    }

    Mat disp8;
    normalize(src16, disp8, 0, 255, NORM_MINMAX);
    disp8.convertTo(disp8, CV_8U);
    cvtColor(disp8, disp8, COLOR_GRAY2BGR);
    draw_points(disp8, pos);

    namedWindow("原图 + 检测点 (ESC退出)", WINDOW_AUTOSIZE);
    imshow("原图 + 检测点 (ESC退出)", disp8);
    cout << "PointRow=" << PointRow << " PointCol=" << PointCol << endl;

    waitKey(0);
    destroyAllWindows();
    return 0;
}
