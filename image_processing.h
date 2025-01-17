#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

using namespace cv;

// 滑动窗口找到亮度最白的长方形区域
Rect findBrightestRectangle(const Mat& grayImg);

// 增强亮度最白的长方形区域并遮罩其他区域
void enhanceAndMask(Mat& grayImg, Mat& enhancedImg, const Rect& brightestRect);

#endif // IMAGE_PROCESSING_H
