#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void alignCirclesToGrid(vector<Vec3f>& circles, vector<Vec3f>& alignedCircles) {
    vector<Point> centers;
    for (const auto& circle : circles) {
        centers.emplace_back(cvRound(circle[0]), cvRound(circle[1]));
    }
    sort(centers.begin(), centers.end(), [](const Point& a, const Point& b) {
        return a.x < b.x;
    });
    vector<vector<Point>> groupedByX;
    for (const auto& center : centers) {
        bool added = false;
        for (auto& group : groupedByX) {
            if (abs(center.x - group.back().x) <= 20) { 
                group.push_back(center);
                added = true;
                break;
            }
        }
        if (!added) {
            groupedByX.push_back({center});
        }
    }
    vector<Vec3f> tempCircles;
    for (auto& group : groupedByX) {
        int sumX = 0;
        int count = 0;
        for (const auto& point : group) {
            sumX += point.x;
            count++;
        }
        int avgX = static_cast<int>(sumX / count + 0.5); 
        for (const auto& point : group) {
            tempCircles.push_back(Vec3f(avgX, point.y, circles[0][2])); 
        }
    }
    vector<vector<Point>> groupedByY;
    for (const auto& circle : tempCircles) {
        Point center(cvRound(circle[0]), cvRound(circle[1]));
        bool added = false;
        for (auto& group : groupedByY) {
            if (abs(center.y - group.back().y) <= 20) { 
                group.push_back(center);
                added = true;
                break;
            }
        }
        if (!added) {
            groupedByY.push_back({center});
        }
    }

    alignedCircles.clear();
    for (auto& group : groupedByY) {
        int sumY = 0;
        int count = 0;
        for (const auto& point : group) {
            sumY += point.y;
            count++;
        }
        int avgY = static_cast<int>(sumY / count + 0.5);
        for (const auto& point : group) {
            alignedCircles.push_back(Vec3f(point.x, avgY, circles[0][2]));
        }
    }
    cout << "对齐后的圆心坐标：" << endl;
    for (const auto& circle : alignedCircles) {
        cout << "(" << cvRound(circle[0]) << ", " << cvRound(circle[1]) << ", " << circle[2] << ")" << endl;
    }
}