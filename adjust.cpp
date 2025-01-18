#include <iostream>
#include "image_processing.h"
#include "gridalignment.h"
#include "adjust.h"
#include "sorting.h"
#include <QApplication>
#include <QFileDialog>

void processGroupedCircles(std::vector<std::vector<cv::Vec3f>>& groupedCircles) {
    for (size_t i = 0; i < groupedCircles.size(); ++i) {
        auto& group = groupedCircles[i];
        if (group.size() <= 1) {
            continue;
        }

        float firstX = group[0][0];
        float sumAdjustedDiff = 0.0f;
        size_t count = 0;

        for (size_t j = 1; j < group.size(); ++j) {
            float currentX = group[j][0];
            float diff = currentX - firstX; 
            float adjustedDiff = diff / j; 
            sumAdjustedDiff += adjustedDiff;
            ++count;
        }
        float averageAdjustedDiff = (count > 0) ? (sumAdjustedDiff / count) : 0.0f;
        for (size_t j = 1; j < group.size(); ++j) {
            float currentX = group[j][0];
            float diff = currentX - firstX;
            group[j][0] = currentX - diff;
        }

        size_t n = group.size(); 
        for (size_t j = 0; j < group.size(); ++j) {
            float adjustmentFactor = (j == 0) ? (averageAdjustedDiff / n) : ((j * averageAdjustedDiff) / (n - 1));
            group[j][0] += adjustmentFactor; 
        }
    }
}

void swapXYInGroupedCircles(std::vector<std::vector<cv::Vec3f>>& groupedCircles) {
    for (auto& group : groupedCircles) {
        for (auto& circle : group) {
            float temp = circle[0]; 
            circle[0] = circle[1];
            circle[1] = temp;
        }
    }
}

void flattenGroupedCircles(const std::vector<std::vector<cv::Vec3f>>& groupedCircles, std::vector<cv::Vec3f>& filteredCircles) {
    filteredCircles.clear();
    for (const auto& group : groupedCircles) {
        filteredCircles.insert(filteredCircles.end(), group.begin(), group.end());
    }
}


void processCircles(std::vector<cv::Vec3f>& filteredCircles) {

    std::vector<std::vector<cv::Vec3f>> groupedCircles;
    alignCircles(filteredCircles, groupedCircles);
    processGroupedCircles(groupedCircles);
    swapXYInGroupedCircles(groupedCircles);
    flattenGroupedCircles(groupedCircles, filteredCircles);

    groupedCircles.clear();

    alignCircles(filteredCircles, groupedCircles);
    processGroupedCircles(groupedCircles);
    swapXYInGroupedCircles(groupedCircles);
    flattenGroupedCircles(groupedCircles, filteredCircles);
}