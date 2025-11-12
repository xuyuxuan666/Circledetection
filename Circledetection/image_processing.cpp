#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const int RECT_WIDTH = 798;
const int RECT_HEIGHT = 660;
const int STEP_SIZE = 2; 

Rect findBrightestRectangle(const Mat& grayImg) {
    int height = grayImg.rows;
    int width = grayImg.cols;

    double maxBrightness = -1;
    Rect brightestRect;

    for (int row = 0; row <= height - RECT_HEIGHT; row += STEP_SIZE) {
        for (int col = 0; col <= width - RECT_WIDTH; col += STEP_SIZE) {
            Rect rect(col, row, RECT_WIDTH, RECT_HEIGHT);
            Mat block = grayImg(rect);
            Scalar avgBrightness = mean(block);
            if (avgBrightness[0] > maxBrightness) {
                maxBrightness = avgBrightness[0];
                brightestRect = rect;
            }
        }
    }
    return brightestRect;
}

void enhanceAndMask(Mat& grayImg, Mat& enhancedImg, const Rect& brightestRect) {
    enhancedImg = Mat::zeros(grayImg.size(), grayImg.type());
    Mat brightestRegion = grayImg(brightestRect);
    int subBlockHeight = brightestRect.height / 6;
    int subBlockWidth = brightestRect.width / 7;

    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 7; ++col) {
            int startRow = row * subBlockHeight;
            int startCol = col * subBlockWidth;

            Rect subBlockRect(startCol, startRow, subBlockWidth, subBlockHeight);
            Mat subBlock = brightestRegion(subBlockRect);
            Mat enhancedSubBlock;
            equalizeHist(subBlock, enhancedSubBlock);
            enhancedSubBlock.copyTo(brightestRegion(subBlockRect));
        }
    }
    brightestRegion.copyTo(enhancedImg(brightestRect));
}