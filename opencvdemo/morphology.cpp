//
//  morphology.cpp
//  opencvdemo
//  MORPH_OPEN – 开运算（Opening operation）
//  开运算是对图像先腐蚀再膨胀，可以排除小团的物体
//  MORPH_CLOSE – 闭运算（Closing operation）
//  闭运算是对图像先膨胀再腐蚀，可以排除小型黑洞：
//  MORPH_GRADIENT -形态学梯度（Morphological gradient）
//  返回图片为膨胀图与腐蚀图之差，可以保留物体的边缘轮廓
//  MORPH_TOPHAT - “顶帽”（“Top hat”）
//  返回图像为原图像与开运算结果图之差
//  MORPH_BLACKHAT - “黑帽”（“Black hat“）
//  返回图片为闭运算结果图与原图像之差
//  Created by ChunmengYang on 2019/10/21.
//  Copyright © 2019 ChunmengYang. All rights reserved.
//

#include "morphology.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

static Mat img, dst;

static int g_nMaxIterationNum = 10;
static int g_nOpenCloseNum = 0;
static int g_nErodeDilateNum = 0;
static int g_nTopBlackHatNum = 0;

static int g_nElementShape = MORPH_RECT;

static void on_OpenClose(int,void *);
static void on_ErodeDilate(int,void *);
static void on_TopBlackHat(int,void *);

void morp() {
    // insert code here...
    
    String path = "/Users/mash5/Downloads/2.jpeg";
    img = imread(path);
    
    namedWindow("open/close", 1);
    namedWindow("erode/dilate",1);
    namedWindow("topblackhat",1);
    
    g_nOpenCloseNum = 10;
    g_nErodeDilateNum = 10;
    g_nTopBlackHatNum = 10;
    
    createTrackbar("迭代值", "open/close", &g_nOpenCloseNum, g_nMaxIterationNum * 2 + 1, on_OpenClose);
    createTrackbar("迭代值", "erode/dilate", &g_nErodeDilateNum, g_nMaxIterationNum * 2 + 1, on_ErodeDilate);
    createTrackbar("迭代值", "topblackhat", &g_nTopBlackHatNum, g_nMaxIterationNum * 2 + 1, on_TopBlackHat);
    
    while (1) {
        int c;
        on_OpenClose(g_nOpenCloseNum, 0);
        on_ErodeDilate(g_nErodeDilateNum, 0);
        on_TopBlackHat(g_nTopBlackHatNum, 0);
        
        c = waitKey(0);
        
        if((char)c == 'q'||(char)c == 27)
            break;
        if((char)c == 49)
            g_nElementShape = MORPH_ELLIPSE;
        else if((char)c == 50)
            g_nElementShape = MORPH_RECT;
        else if((char)c == 51)
            g_nElementShape = MORPH_CROSS;
    }
}

// 形态学变换
static void on_OpenClose(int, void *){
    int offset = g_nOpenCloseNum - g_nMaxIterationNum;
    int Absolute_offset = offset > 0 ? offset : -offset;
    /*
     getStructuringElement函数会返回指定形状和尺寸的结构元素（内核矩阵）。
     函数原型：
     Mat getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1));
     参数说明：
     shape：表示内核的形状
                   MORPH_RECT， 矩形；
                   MORPH_CROSS，交叉形；
                   MORPH_ELLIPSE，椭圆形；
     ksize：内核的尺寸；
     anchor ：锚点的位置。
     */
    Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), Point(Absolute_offset, Absolute_offset));
    
    /*
     void morphologyEx(InputArray src, OutputArray dst, int op, InputArray kernel, Point anchor=Point(-1,-1), intiterations=1, int borderType=BORDER_CONSTANT, const Scalar& borderValue=morphologyDefaultBorderValue() )
     函数参数：
     第一个参数，输入图像
     第二个参数，输出图像
     第三个参数，使用的形态学方法即：
     MORPH_OPEN – 开运算（Opening operation）
     开运算是对图像先腐蚀再膨胀，可以排除小团的物体
        MORPH_CLOSE – 闭运算（Closing operation）
     闭运算是对图像先膨胀再腐蚀，可以排除小型黑洞：
        MORPH_GRADIENT -形态学梯度（Morphological gradient）
     返回图片为膨胀图与腐蚀图之差，可以保留物体的边缘轮廓
        MORPH_TOPHAT - “顶帽”（“Top hat”）
     返回图像为原图像与开运算结果图之差
        MORPH_BLACKHAT - “黑帽”（“Black hat“）
     返回图片为闭运算结果图与原图像之差
     第四个参数，InputArray类型的kernel，形态学运算的内核。若为NULL时，表示的是使用参考点位于中心3x3的核。如果设置5*5的即：Mat(5, 5, CV_8U)
     第五个参数，Point类型的anchor，锚的位置，其有默认值（-1，-1），表示锚位于中心。
     第六个参数，int类型的iterations，迭代使用函数的次数，默认值为1。
     第七个参数，int类型的borderType，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_ CONSTANT。
     第八个参数，const Scalar&类型的borderValue，当边界为常数时的边界值，有默认值morphologyDefaultBorderValue()，
     */
    
    if (offset < 0) {
        morphologyEx(img, dst, MORPH_OPEN, element);
    } else {
        morphologyEx(img, dst, MORPH_CLOSE, element);
    }
    
    imshow("open/close", dst);
}


static void on_ErodeDilate(int,void *){
    int offset = g_nErodeDilateNum-g_nMaxIterationNum;
    int Absolute_offset = offset > 0 ? offset : -offset;
    
    Mat element=getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1,Absolute_offset * 2 + 1),Point(Absolute_offset, Absolute_offset));
    
    if (offset < 0) {
        erode(img, dst, element);
    } else {
        dilate(img, dst, element);
    }

    imshow("erode/dilate", dst);
}



static void on_TopBlackHat(int,void *){
    int offset = g_nTopBlackHatNum - g_nMaxIterationNum;
    int Absolute_offset = offset > 0 ? offset : -offset;
   
    Mat element=getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1,Absolute_offset * 2 + 1),Point(Absolute_offset, Absolute_offset));
    
    if (offset < 0) {
        morphologyEx(img, dst, MORPH_TOPHAT, element);
    } else {
        morphologyEx(img, dst, MORPH_BLACKHAT, element);
    }
    
    imshow("topblackhat", dst);
}
