#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <cmath>

using namespace std;
using namespace cv;

void alignCircles(vector<Vec3f>& circles, vector<vector<Vec3f>>& groupedCircles, int threshold = 30) {
    sort(circles.begin(), circles.end(), [](const Vec3f& a, const Vec3f& b) {
        return a[0] < b[0];
    });

    vector<Vec3f> currentGroup;
    for (size_t i = 0; i < circles.size(); ++i) {
        if (currentGroup.empty() || abs(circles[i][0] - currentGroup.back()[0]) <= threshold) {
            currentGroup.push_back(circles[i]);
        } else {
            groupedCircles.push_back(currentGroup);
            currentGroup.clear();
            currentGroup.push_back(circles[i]);
        }
    }
    if (!currentGroup.empty()) {
        groupedCircles.push_back(currentGroup);
    }
    for (auto& group : groupedCircles) {
        sort(group.begin(), group.end(), [](const Vec3f& a, const Vec3f& b) {
            return a[1] < b[1]; 
        });
    }
    // cout << "Grouped Circles (sorted by y-axis within groups):" << endl;
    for (size_t i = 0; i < groupedCircles.size(); ++i) {
        // cout << "Group " << i + 1 << ": ";
        for (const auto& circle : groupedCircles[i]) {
            // cout << "(" << circle[0] << ", " << circle[1] << ", " << circle[2] << ") ";
        }
        // cout << endl;
    }
}
