#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const int BLOCK_ROWS = 8;
const int BLOCK_COLS = 10;
const int FILTER_SIZE = 7; 
const int MIN_CONTOUR_AREA = 400;

void processBlock(Mat& input, Mat& output, Rect blockRect) {
    Mat block = input(blockRect);

    Mat enhancedBlock;
    equalizeHist(block, enhancedBlock);

    Mat filteredBlock;
    medianBlur(enhancedBlock, filteredBlock, FILTER_SIZE);

    filteredBlock.copyTo(output(blockRect)); 
}

void detectContoursCenters(const Mat& img, Mat& resultImg) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    int n = 0; 
    for (size_t i = 0; i < contours.size(); ++i) {

        if (contourArea(contours[i]) > MIN_CONTOUR_AREA) {
            Moments m = moments(contours[i]);
            Point center(cvRound(m.m10 / m.m00), cvRound(m.m01 / m.m00));

            cout << "轮廓 " << n + 1 << ": " << endl; 
            cout << "中心坐标: (" << center.x << ", " << center.y << ")" << endl;
            cout << "-------------------------" << endl;

            circle(resultImg, center, 5, Scalar(0, 255, 0), -1);
            n++;
        }
    }
}

int main() {

    Mat img = imread("20250109-02.png"); 
    if (img.empty()) {
        cerr << "图片加载失败！" << endl;
        return -1;
    }

    imshow("原始图像", img);

    Mat grayImg;
    if (img.channels() == 3) {
        cvtColor(img, grayImg, COLOR_BGR2GRAY);
    } else {
        grayImg = img.clone();
    }

    int height = grayImg.rows;
    int width = grayImg.cols;

    int blockHeight = height / BLOCK_ROWS;
    int blockWidth = width / BLOCK_COLS;

    Mat enhancedImg = Mat::zeros(grayImg.size(), grayImg.type());
    Mat binaryImg = Mat::zeros(grayImg.size(), CV_8U);

    for (int row = 0; row < BLOCK_ROWS; ++row) {
        for (int col = 0; col < BLOCK_COLS; ++col) {
            int startRow = row * blockHeight;
            int endRow = min((row + 1) * blockHeight, height);
            int startCol = col * blockWidth;
            int endCol = min((col + 1) * blockWidth, width);

            Rect blockRect(startCol, startRow, endCol - startCol, endRow - startRow);
            processBlock(grayImg, enhancedImg, blockRect);
        }
    }
    threshold(enhancedImg, binaryImg, 80, 255, THRESH_BINARY);

    Mat dilatedImg;
    Mat se = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    dilate(binaryImg, dilatedImg, se);

    Mat closedImg;
    morphologyEx(dilatedImg, closedImg, MORPH_CLOSE, se);

    Mat openedImg;
    morphologyEx(closedImg, openedImg, MORPH_OPEN, se);

    Mat edgeImg;
    Canny(openedImg, edgeImg, 50, 150);
    imshow("Canny 边缘检测结果", edgeImg);

    Mat resultImg = img.clone();

    detectContoursCenters(edgeImg, resultImg);

    imshow("检测到的轮廓中心", resultImg);

    waitKey(0);
    return 0;
}
