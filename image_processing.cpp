#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 定义长方形边长和滑动步长（可根据需求调整）
const int RECT_WIDTH = 798;
const int RECT_HEIGHT = 660;
const int STEP_SIZE = 2; // 滑动步长

// 在图像上框选出亮度最白的长方形区域
Rect findBrightestRectangle(const Mat& grayImg) {
    int height = grayImg.rows;
    int width = grayImg.cols;

    double maxBrightness = -1;
    Rect brightestRect;

    // 滑动窗口遍历图像
    for (int row = 0; row <= height - RECT_HEIGHT; row += STEP_SIZE) {
        for (int col = 0; col <= width - RECT_WIDTH; col += STEP_SIZE) {
            Rect rect(col, row, RECT_WIDTH, RECT_HEIGHT);
            Mat block = grayImg(rect);

            // 计算当前块的平均亮度
            Scalar avgBrightness = mean(block);
            if (avgBrightness[0] > maxBrightness) {
                maxBrightness = avgBrightness[0];
                brightestRect = rect;
            }
        }
    }
    return brightestRect;
}

// 增强亮度最白的长方形区域并遮罩其他区域
void enhanceAndMask(Mat& grayImg, Mat& enhancedImg, const Rect& brightestRect) {
    // 创建一个黑色图像
    enhancedImg = Mat::zeros(grayImg.size(), grayImg.type());

    // 获取最亮区域
    Mat brightestRegion = grayImg(brightestRect);

    // 划分为 6 行 7 列的小块并进行对比度增强
    int subBlockHeight = brightestRect.height / 6;
    int subBlockWidth = brightestRect.width / 7;

    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 7; ++col) {
            int startRow = row * subBlockHeight;
            int startCol = col * subBlockWidth;

            Rect subBlockRect(startCol, startRow, subBlockWidth, subBlockHeight);
            Mat subBlock = brightestRegion(subBlockRect);

            // 增强对比度
            Mat enhancedSubBlock;
            equalizeHist(subBlock, enhancedSubBlock);

            // 将增强后的小块拷贝回对应位置
            enhancedSubBlock.copyTo(brightestRegion(subBlockRect));
        }
    }

    // 将增强后的区域拷贝到输出图像
    brightestRegion.copyTo(enhancedImg(brightestRect));
}