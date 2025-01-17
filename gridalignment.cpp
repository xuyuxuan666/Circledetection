#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void alignCirclesToGrid(vector<Vec3f>& circles, vector<Vec3f>& alignedCircles) {
    // 提取圆心坐标
    vector<Point> centers;
    for (const auto& circle : circles) {
        centers.emplace_back(cvRound(circle[0]), cvRound(circle[1]));
    }

    // 按 x 坐标排序
    sort(centers.begin(), centers.end(), [](const Point& a, const Point& b) {
        return a.x < b.x;
    });

    // 按 x 坐标将圆心分组，x 数值相差不超过 20
    vector<vector<Point>> groupedByX;
    for (const auto& center : centers) {
        bool added = false;
        for (auto& group : groupedByX) {
            if (abs(center.x - group.back().x) <= 20) { // x 方向接近的圆心分组
                group.push_back(center);
                added = true;
                break;
            }
        }
        if (!added) {
            groupedByX.push_back({center});
        }
    }

    // 对每个 x 组的圆心，取平均值并保留 y 和半径不变
    vector<Vec3f> tempCircles;
    for (auto& group : groupedByX) {
        int sumX = 0;
        int count = 0;
        for (const auto& point : group) {
            sumX += point.x;
            count++;
        }
        int avgX = static_cast<int>(sumX / count + 0.5); // 计算平均 x 坐标并四舍五入

        // 保留 y 和半径不变
        for (const auto& point : group) {
            tempCircles.push_back(Vec3f(avgX, point.y, circles[0][2])); // 保留半径
        }
    }

    // 处理 y 坐标，y 数值相差不超过 20 的圆心归为一组，取平均值
    vector<vector<Point>> groupedByY;
    for (const auto& circle : tempCircles) {
        Point center(cvRound(circle[0]), cvRound(circle[1]));
        bool added = false;
        for (auto& group : groupedByY) {
            if (abs(center.y - group.back().y) <= 20) { // y 方向接近的圆心分组
                group.push_back(center);
                added = true;
                break;
            }
        }
        if (!added) {
            groupedByY.push_back({center});
        }
    }

    // 对每个 y 组的圆心，取平均值并保留 x 和半径不变
    alignedCircles.clear();
    for (auto& group : groupedByY) {
        int sumY = 0;
        int count = 0;
        for (const auto& point : group) {
            sumY += point.y;
            count++;
        }
        int avgY = static_cast<int>(sumY / count + 0.5); // 计算平均 y 坐标并四舍五入

        // 保留 x 和半径不变
        for (const auto& point : group) {
            alignedCircles.push_back(Vec3f(point.x, avgY, circles[0][2])); // 保留半径
        }
    }

    // 打印新的对齐圆心数据
    cout << "对齐后的圆心坐标：" << endl;
    for (const auto& circle : alignedCircles) {
        cout << "(" << cvRound(circle[0]) << ", " << cvRound(circle[1]) << ", " << circle[2] << ")" << endl;
    }
}