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

    // 打开文件选择对话框
    QString fileName = QFileDialog::getOpenFileName(
        nullptr,
        "选择图片",
        "",
        "Images (*.bmp *.jpg *.jpeg *.png *.tif);;All Files (*)");

    if (fileName.isEmpty()) {
        std::cerr << "未选择任何图片！" << std::endl;
        return -1;
    }

    // 加载图片
    cv::Mat img = cv::imread(fileName.toStdString());
    if (img.empty()) {
        std::cerr << "无法加载图片: " << fileName.toStdString() << std::endl;
        return -1;
    }

    imshow("原始图像", img);

    // 转为灰度图像
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    // 整体直方图均衡化（整体增强）
    Mat enhancedGrayImg;
    equalizeHist(grayImg, enhancedGrayImg);
    // 使用滑动窗口找到最白的长方形区域
    Rect brightestRect = findBrightestRectangle(enhancedGrayImg);

    // 显示最白区域
    Mat highlightedImg = img.clone();
    rectangle(highlightedImg, brightestRect, Scalar(0, 0, 255), 2); // 红色框
    // 增强亮度最白的长方形区域并遮罩其他区域
    Mat enhancedImg;
    enhanceAndMask(enhancedGrayImg, enhancedImg, brightestRect);
    imshow("增强并遮罩后的图像", enhancedImg);

    Mat colorImg;
    cv::cvtColor(enhancedImg, colorImg, cv::COLOR_GRAY2BGR);


    // 使用中值滤波去除噪声点
    Mat denoisedImg;
    medianBlur(enhancedImg, denoisedImg, 3);  // 3x3 的中值滤波

    // 二值化
    Mat binaryImg;
    threshold(denoisedImg, binaryImg, 140, 255, THRESH_BINARY);

    // Canny边缘检测
    Mat edges;
    Canny(binaryImg, edges, 100, 200);  // Canny边缘检测，100和200是低阈值和高阈值
    imshow("Canny边缘检测", edges);

    // 创建结构元素（3x3矩形）
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    // 形态学膨胀操作
    Mat dilatedImg;
    dilate(edges, dilatedImg, kernel);  // 膨胀操作
    imshow("膨胀后的图像", dilatedImg);

    // // 形态学腐蚀操作
    // Mat erodedImg;
    // erode(edges, erodedImg, kernel);  // 腐蚀操作
    // imshow("腐蚀后的图像", erodedImg);

    // // 形态学闭操作：填补空洞
    // Mat closedImg;
    // morphologyEx(edges, closedImg, MORPH_CLOSE, kernel);  // 闭操作
    // imshow("闭操作后的图像", closedImg);

    // // 形态学开操作：去除小噪点
    // Mat openedImg;
    // morphologyEx(edges, openedImg, MORPH_OPEN, kernel);  // 开操作
    // imshow("开操作后的图像", openedImg);



    ///////////////////////////////////////////////////////////////////
    // Hough圆检测，半径大于40小于70
    vector<Vec3f> detectedCircles;
    HoughCircles(dilatedImg, detectedCircles, HOUGH_GRADIENT, 1, 85, 30, 20, 40, 70);



    // 绘制检测到的圆
    Mat circleImg = colorImg.clone();  // 使用原图作为底图

    if (detectedCircles.empty()) {
        cout << "未检测到圆。" << endl;
    } else {
        vector<Vec3f> filteredCircles;
        
        // 过滤圆心间隔大于40的圆
        for (size_t i = 0; i < detectedCircles.size(); ++i) {
            bool tooClose = false;
            Point center(cvRound(detectedCircles[i][0]), cvRound(detectedCircles[i][1]));
            int radius = cvRound(detectedCircles[i][2]);

            // 检查当前圆是否与其他圆心间隔小于40
            for (size_t j = 0; j < filteredCircles.size(); ++j) {
                Point otherCenter(cvRound(filteredCircles[j][0]), cvRound(filteredCircles[j][1]));
                double dist = norm(center - otherCenter);  // 计算圆心之间的距离
                if (dist < 40) {
                    tooClose = true;  // 如果距离小于40，则标记为太近
                    break;
                }
            }

            // 如果圆心间隔大于40，则将圆添加到过滤后的圆列表
            if (!tooClose) {
                filteredCircles.push_back(detectedCircles[i]);
            }
        }

        // // 打印 filteredCircles
        // for (const auto& circle : filteredCircles) {
        //     cout << "Center: (" << circle[0] << ", " << circle[1] << "), Radius: " << circle[2] << endl;
        // }
        // // 对齐圆心到网格
        // cout << "xxxxxxxxxxxx" << endl;
        // alignCirclesToGrid(filteredCircles, filteredCircles);
        // for (const auto& circle : filteredCircles) {
        //     cout << "Center: (" << circle[0] << ", " << circle[1] << "), Radius: " << circle[2] << endl;
        // }
        // alignCircles(filteredCircles);
        std::vector<std::vector<cv::Vec3f>> groupedCircles;

        alignCircles(filteredCircles, groupedCircles);



        // // 打印 groupedCircles
        // std::cout << "Grouped Circles after alignment and sorting:" << std::endl;
        // for (size_t i = 0; i < groupedCircles.size(); ++i) {
        //     std::cout << "Group " << i + 1 << ": ";
        //     for (const auto& circle : groupedCircles[i]) {
        //         std::cout << "(" << circle[0] << ", " << circle[1] << ", " << circle[2] << ") ";
        //     }
        //     std::cout << std::endl;
        // }


        processGroupedCircles(groupedCircles);
        swapXYInGroupedCircles(groupedCircles);
        flattenGroupedCircles(groupedCircles,filteredCircles);
        alignCircles(filteredCircles, groupedCircles);
        processGroupedCircles(groupedCircles);
        swapXYInGroupedCircles(groupedCircles);
        flattenGroupedCircles(groupedCircles,filteredCircles);


        // 如果没有符合条件的圆
        if (filteredCircles.empty()) {
            cout << "未检测到符合条件的圆。" << endl;
        } else {
        
            for (size_t i = 0; i < filteredCircles.size(); ++i) {
                Vec3f detectedCircle = filteredCircles[i];
                Point center(cvRound(detectedCircle[0]), cvRound(detectedCircle[1]));
                int radius = cvRound(detectedCircle[2]);

                // 绘制圆心为红色十字
                line(circleImg, Point(center.x - 10, center.y), Point(center.x + 10, center.y), Scalar(0, 0, 255), 2);
                line(circleImg, Point(center.x, center.y - 10), Point(center.x, center.y + 10), Scalar(0, 0, 255), 2);

                // 绘制圆轮廓
                circle(circleImg, center, 27, Scalar(0, 255, 0), 2);

                // 打印圆信息
                cout << "圆心: (" << center.x << ", " << center.y << "), 半径: " << radius << endl;
            }
        }
    }

    imshow("检测到的圆", circleImg);
    //////////////////////////////////////////////////////////////////////////////////////

    waitKey(0);
    return 0;
}
