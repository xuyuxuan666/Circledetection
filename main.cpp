#include <iostream>
#include "image_processing.h"
#include "gridalignment.h"
#include "adjust.h"
#include "sorting.h"
#include <QApplication>
#include <QFileDialog>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QString fileName = QFileDialog::getOpenFileName(
        nullptr,
        "选择图片",
        "",
        "Images (*.bmp *.jpg *.jpeg *.png *.tif);;All Files (*)");

    if (fileName.isEmpty()) {
        std::cerr << "未选择任何图片！" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(fileName.toStdString());
    if (img.empty()) {
        std::cerr << "无法加载图片: " << fileName.toStdString() << std::endl;
        return -1;
    }

    imshow("原始图像", img);

    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    Mat enhancedGrayImg;
    equalizeHist(grayImg, enhancedGrayImg);
    Rect brightestRect = findBrightestRectangle(enhancedGrayImg);

    Mat highlightedImg = img.clone();
    rectangle(highlightedImg, brightestRect, Scalar(0, 0, 255), 2); 
    Mat enhancedImg;
    enhanceAndMask(enhancedGrayImg, enhancedImg, brightestRect);
    Mat colorImg;
    cv::cvtColor(enhancedImg, colorImg, cv::COLOR_GRAY2BGR);
    Mat denoisedImg;
    medianBlur(enhancedImg, denoisedImg, 3); 

    Mat binaryImg;
    threshold(denoisedImg, binaryImg, 120, 255, THRESH_BINARY);


    Mat edges;
    Canny(binaryImg, edges, 100, 200); 

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat dilatedImg;
    dilate(edges, dilatedImg, kernel);

    vector<Vec3f> detectedCircles;
    HoughCircles(dilatedImg, detectedCircles, HOUGH_GRADIENT, 1, 85, 30, 20, 40, 70);

    Mat circleImg = colorImg.clone(); 

    if (detectedCircles.empty()) {
        cout << "未检测到圆。" << endl;
    } else {
        vector<Vec3f> filteredCircles;
        for (size_t i = 0; i < detectedCircles.size(); ++i) {
            bool tooClose = false;
            Point center(cvRound(detectedCircles[i][0]), cvRound(detectedCircles[i][1]));
            int radius = cvRound(detectedCircles[i][2]);
            for (size_t j = 0; j < filteredCircles.size(); ++j) {
                Point otherCenter(cvRound(filteredCircles[j][0]), cvRound(filteredCircles[j][1]));
                double dist = norm(center - otherCenter); 
                if (dist < 40) {
                    tooClose = true; 
                    break;
                }
            }
            if (!tooClose) {
                filteredCircles.push_back(detectedCircles[i]);
            }
        }
        // 调整方法一 
        // alignCirclesToGrid(filteredCircles,filteredCircles);
        // 调整方法二
        processCircles(filteredCircles);

        if (filteredCircles.empty()) {
            cout << "未检测到符合条件的圆。" << endl;
        } else {
            int label = 1;
            for (size_t i = 0; i < filteredCircles.size(); ++i) {
                Vec3f detectedCircle = filteredCircles[i];
                Point center(cvRound(detectedCircle[0]), cvRound(detectedCircle[1]));
                int radius = cvRound(detectedCircle[2]);

                line(circleImg, Point(center.x - 10, center.y), Point(center.x + 10, center.y), Scalar(0, 0, 255), 2);
                line(circleImg, Point(center.x, center.y - 10), Point(center.x, center.y + 10), Scalar(0, 0, 255), 2);

                circle(circleImg, center, 27, Scalar(0, 255, 0), 2);

                cout << "圆心: (" << center.x << ", " << center.y << ")" << ", 编号: " << label << endl;
                label=label+1;
            }
        }
    }

    imshow("检测到的圆", circleImg);

    waitKey(0);
    return 0;
}
