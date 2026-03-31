// #include <iostream>
// #include <vector>
// #include <opencv2/opencv.hpp>

// void processGroupedCircles(const std::vector<std::vector<cv::Vec3f>>& groupedCircles) {
//     // 遍历每组
//     for (size_t i = 0; i < groupedCircles.size(); ++i) {
//         const auto& group = groupedCircles[i]; // 当前组
//         if (group.size() <= 1) {
//             // 如果组中点数少于等于1，则跳过
//             continue;
//         }

//         std::cout << "Processing Group " << i + 1 << ":\n";
//         float firstX = group[0][0]; // 第一点的 x 值
//         float sumAdjustedDiff = 0.0f; // 累积 adjustedDiff
//         size_t count = 0; // 用于计算平均值的计数

//         // 遍历组中的点，从第二个点开始
//         for (size_t j = 1; j < group.size(); ++j) {
//             float currentX = group[j][0]; // 当前点的 x 值
//             float diff = currentX - firstX; // 与第一个点的差值
//             float adjustedDiff = diff / j; // 差值除以索引 j
//             sumAdjustedDiff += adjustedDiff; // 累加 adjustedDiff
//             ++count; // 更新计数
//             std::cout << "Point " << j + 1 << ": x = " << currentX 
//                       << ", diff = " << diff 
//                       << ", adjustedDiff = " << adjustedDiff << "\n";
//         }

//         // 计算平均 adjustedDiff
//         float averageAdjustedDiff = (count > 0) ? (sumAdjustedDiff / count) : 0.0f;
//         std::cout << "Average adjustedDiff for Group " << i + 1 << ": " 
//                   << averageAdjustedDiff << "\n\n";
//     }
// }




#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


void processGroupedCircles(std::vector<std::vector<cv::Vec3f>>& groupedCircles) {
    // 遍历每组
    for (size_t i = 0; i < groupedCircles.size(); ++i) {
        auto& group = groupedCircles[i]; // 当前组，直接修改 groupedCircles[i]
        if (group.size() <= 1) {
            // 如果组中点数少于等于1，则跳过
            continue;
        }

        std::cout << "Processing Group " << i + 1 << ":\n";
        float firstX = group[0][0]; // 第一点的 x 值
        float sumAdjustedDiff = 0.0f; // 累积 adjustedDiff
        size_t count = 0; // 用于计算平均值的计数

        // 遍历组中的点，从第二个点开始
        for (size_t j = 1; j < group.size(); ++j) {
            float currentX = group[j][0]; // 当前点的 x 值
            float diff = currentX - firstX; // 与第一个点的差值
            float adjustedDiff = diff / j; // 差值除以索引 j
            sumAdjustedDiff += adjustedDiff; // 累加 adjustedDiff
            ++count; // 更新计数
            std::cout << "Point " << j + 1 << ": x = " << currentX 
                      << ", diff = " << diff 
                      << ", adjustedDiff = " << adjustedDiff << "\n";
        }

        // 计算平均 adjustedDiff
        float averageAdjustedDiff = (count > 0) ? (sumAdjustedDiff / count) : 0.0f;
        std::cout << "Average adjustedDiff for Group " << i + 1 << ": " 
                  << averageAdjustedDiff << "\n";

        // 第一部：每组的每个点减去自身的diff
        for (size_t j = 1; j < group.size(); ++j) {
            float currentX = group[j][0];
            float diff = currentX - firstX;
            group[j][0] = currentX - diff; // 将 x 坐标减去自身的 diff
            std::cout << "Adjusted Point " << j + 1 << ": x = " << group[j][0] << "\n";
        }

        // 第二部：根据 averageAdjustedDiff 进行调整
        size_t n = group.size(); // 每组点的数量
        for (size_t j = 0; j < group.size(); ++j) {
            float adjustmentFactor = (j == 0) ? (averageAdjustedDiff / n) : ((j * averageAdjustedDiff) / (n - 1));
            group[j][0] += adjustmentFactor; // 根据位置调整每个点
            std::cout << "Adjusted Point " << j + 1 << ": x = " << group[j][0] << ", adjustmentFactor = " << adjustmentFactor << "\n";
        }

        std::cout << "\n";
    }
}



void swapXYInGroupedCircles(std::vector<std::vector<cv::Vec3f>>& groupedCircles) {
    for (auto& group : groupedCircles) {
        for (auto& circle : group) {
            // 交换每个点的 x 和 y 坐标
            float temp = circle[0]; // 保存 x 坐标
            circle[0] = circle[1]; // 将 y 坐标赋给 x
            circle[1] = temp; // 将保存的 x 坐标赋给 y
        }
    }
}

void flattenGroupedCircles(const std::vector<std::vector<cv::Vec3f>>& groupedCircles, std::vector<cv::Vec3f>& filteredCircles) {
    // 清空 filteredCircles
    filteredCircles.clear();

    // 遍历每一组，将每一组的元素添加到 filteredCircles 中
    for (const auto& group : groupedCircles) {
        filteredCircles.insert(filteredCircles.end(), group.begin(), group.end());
    }
}
