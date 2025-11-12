#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

using namespace cv;

Rect findBrightestRectangle(const Mat& grayImg);
void enhanceAndMask(Mat& grayImg, Mat& enhancedImg, const Rect& brightestRect);

#endif 
