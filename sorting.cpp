#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <cmath>

using namespace std;
using namespace cv;

// // 计算线性拟合的斜率
// double computeSlope(const vector<double>& x, const vector<double>& y) {
//     int n = x.size();
//     if (n < 2) return 0.0;

//     double x_mean = accumulate(x.begin(), x.end(), 0.0) / n;
//     double y_mean = accumulate(y.begin(), y.end(), 0.0) / n;

//     double numerator = 0.0, denominator = 0.0;
//     for (int i = 0; i < n; ++i) {
//         numerator += (x[i] - x_mean) * (y[i] - y_mean);
//         denominator += (x[i] - x_mean) * (x[i] - x_mean);
//     }

//     return denominator == 0 ? 0.0 : numerator / denominator;
// }

// // 根据指定轴对圆心分组
// vector<vector<int>> groupCirclesByAxis(const vector<Vec3f>& circles, char axis) {
//     double threshold = 1.0; // 设置默认的阈值为 1.0
//     vector<vector<int>> groups;
//     vector<bool> visited(circles.size(), false);

//     for (size_t i = 0; i < circles.size(); ++i) {
//         if (visited[i]) continue;

//         vector<int> group;
//         group.push_back(i);
//         visited[i] = true;

//         double reference = (axis == 'x') ? circles[i][0] : circles[i][1];

//         for (size_t j = i + 1; j < circles.size(); ++j) {
//             if (visited[j]) continue;

//             double value = (axis == 'x') ? circles[j][0] : circles[j][1];
//             if (fabs(value - reference) <= threshold) {
//                 group.push_back(j);
//                 visited[j] = true;
//             }
//         }

//         groups.push_back(group);
//     }

//     return groups;
// }

// // 调整圆心位置使其落在拟合直线上
// void adjustCirclePositions(vector<Vec3f>& circles, const vector<vector<int>>& groups, char axis) {
//     for (const auto& group : groups) {
//         vector<double> x_vals, y_vals;
//         for (int idx : group) {
//             x_vals.push_back(circles[idx][0]);
//             y_vals.push_back(circles[idx][1]);
//         }

//         double slope = computeSlope(x_vals, y_vals);
//         double x_mean = accumulate(x_vals.begin(), x_vals.end(), 0.0) / x_vals.size();
//         double y_mean = accumulate(y_vals.begin(), y_vals.end(), 0.0) / y_vals.size();

//         for (int idx : group) {
//             if (axis == 'x') {
//                 // 修改圆心的 y 坐标，使其落在拟合的直线上
//                 circles[idx][1] = slope * (circles[idx][0] - x_mean) + y_mean;
//             } else if (axis == 'y') {
//                 // 修改圆心的 x 坐标，使其落在拟合的直线上
//                 circles[idx][0] = (circles[idx][1] - y_mean) / slope + x_mean;
//             }
//         }
//     }
// }

// // 调整输入圆心位置的主函数
// void alignCircles(vector<Vec3f>& circles) {
//     // 通过按 x 轴分组并调整圆心
//     vector<vector<int>> x_groups = groupCirclesByAxis(circles, 'x');
//     adjustCirclePositions(circles, x_groups, 'x');

// }

/*
//  x fenzu
void alignCircles(vector<Vec3f>& circles, vector<vector<Vec3f>>& groupedCircles, int threshold = 30) {
    // 按照 x 坐标对圆心排序
    sort(circles.begin(), circles.end(), [](const Vec3f& a, const Vec3f& b) {
        return a[0] < b[0];
    });

    // 分组
    vector<Vec3f> currentGroup;
    for (size_t i = 0; i < circles.size(); ++i) {
        if (currentGroup.empty() || abs(circles[i][0] - currentGroup.back()[0]) <= threshold) {
            // 如果当前分组为空，或者 x 轴差值小于等于阈值，则加入当前组
            currentGroup.push_back(circles[i]);
        } else {
            // 否则，完成当前组，将其加入分组结果
            groupedCircles.push_back(currentGroup);
            currentGroup.clear();
            currentGroup.push_back(circles[i]);
        }
    }
    // 将最后的分组加入结果
    if (!currentGroup.empty()) {
        groupedCircles.push_back(currentGroup);
    }

    // 打印分组结果
    cout << "Grouped Circles:" << endl;
    for (size_t i = 0; i < groupedCircles.size(); ++i) {
        cout << "Group " << i + 1 << ": ";
        for (const auto& circle : groupedCircles[i]) {
            cout << "(" << circle[0] << ", " << circle[1] << ", " << circle[2] << ") ";
        }
        cout << endl;
    }
}
*/

void alignCircles(vector<Vec3f>& circles, vector<vector<Vec3f>>& groupedCircles, int threshold = 30) {
    // 按照 x 坐标对圆心排序
    sort(circles.begin(), circles.end(), [](const Vec3f& a, const Vec3f& b) {
        return a[0] < b[0];
    });

    // 分组
    vector<Vec3f> currentGroup;
    for (size_t i = 0; i < circles.size(); ++i) {
        if (currentGroup.empty() || abs(circles[i][0] - currentGroup.back()[0]) <= threshold) {
            // 如果当前分组为空，或者 x 轴差值小于等于阈值，则加入当前组
            currentGroup.push_back(circles[i]);
        } else {
            // 否则，完成当前组，将其加入分组结果
            groupedCircles.push_back(currentGroup);
            currentGroup.clear();
            currentGroup.push_back(circles[i]);
        }
    }
    // 将最后的分组加入结果
    if (!currentGroup.empty()) {
        groupedCircles.push_back(currentGroup);
    }

    // 按照 y 轴排序每一组
    for (auto& group : groupedCircles) {
        sort(group.begin(), group.end(), [](const Vec3f& a, const Vec3f& b) {
            return a[1] < b[1]; // 按照 y 坐标排序
        });
    }

    // 打印分组结果
    cout << "Grouped Circles (sorted by y-axis within groups):" << endl;
    for (size_t i = 0; i < groupedCircles.size(); ++i) {
        cout << "Group " << i + 1 << ": ";
        for (const auto& circle : groupedCircles[i]) {
            cout << "(" << circle[0] << ", " << circle[1] << ", " << circle[2] << ") ";
        }
        cout << endl;
    }
}
