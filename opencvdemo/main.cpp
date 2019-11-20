//
//  main.cpp
//  opencvdemo
//
//  Created by ChunmengYang on 2019/10/16.
//  Copyright © 2019 ChunmengYang. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "morphology.hpp"

using namespace cv;
using namespace std;

static void houghLinesDemo();
static void cannyDemo();
static void laplacianDemo();
static void scharrDemo();
static void gaussianDiff();
static Mat zoomout(Mat &src);
static void thresholdDemo();
static void linearBlend();
static void roiCopy();
static void grayDemo();
static void drawDemo();
static void pixelOperation();
static void blurDemo();
static void erodeDemo();
static void dilateDemo();
static void contrastDemo();


int main(int argc, const char * argv[]) {
    // insert code here...
//    String path = "/Users/mash5/Downloads/12.jfif";
//
//    Mat img = imread(path);
//    imshow("Image", img);
//    imwrite("/Users/mash5/Downloads/10-6.jpg", img);
//    
//    Mat img_gray;
//    cvtColor(img, img_gray, COLOR_BGR2GRAY);
//    imshow("Image Gray", img_gray);
//    imwrite("/Users/mash5/Downloads/10-4.jpg", img_gray);
   
//    // 二值化图像
//    Mat img_binary;
//    threshold(img_gray, img_binary, 0, 255, THRESH_OTSU | THRESH_BINARY);
//    imshow("Image Binary;", img_binary);
//    imwrite("/Users/mash5/Downloads/10-4.jpg", img_binary);
    
    // 归一化数据
//    Mat dst;
//    normalize(img_gray, dst, 150, 255, NORM_MINMAX);
//    imshow("Image Binary", dst);
//    imwrite("/Users/mash5/Downloads/10-5.jpg", dst);
    
//    waitKey(0);
//
//    Mat dst;
//    adaptiveThreshold(img_gray, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 5);
//    imshow("adaptiveThreshold", dst);
//
//    Mat wline = getStructuringElement(MORPH_RECT, Size(dst.cols / 16, 1), Point(-1, -1));
//    Mat hline = getStructuringElement(MORPH_RECT, Size(1, dst.rows / 16), Point(-1, -1));
//
//    Mat temp;
//    erode(dst, temp, wline);
//    imshow("wline", temp);
//
//    dilate(temp, dst, hline);
//    bitwise_not(dst, dst);
//    imshow("hline", dst);
//    waitKey(0);
    
    Mat gray;//当前帧灰度图
    Mat background;//背景图，格式为32位浮点
    Mat backImage;//CV_8U格式背景图
    Mat foreground;//前景图
    Mat output;
    double learningRate = 0.5;//学习率
    int threshold = 30;//阈值，滤去扰动
    
    VideoCapture capture;
    capture.open(0);
    while (capture.isOpened()) {
        Mat frame;
        capture >> frame;
        cvtColor(frame, gray, CV_BGR2GRAY);
        if (background.empty()) {
            gray.convertTo(background, CV_32F);
        }
        background.convertTo(backImage, CV_8U);
        absdiff(backImage, gray, foreground);
        
        cv::threshold(foreground, output, threshold, 255, THRESH_BINARY_INV);
        accumulateWeighted(gray, background, learningRate);
        
//        imshow("frame", backImage);
        imshow("result", output);

        int key = waitKey(10);
        if (key == 27) {
            break;
        }
    }

    return 0;
}

static void houghLinesDemo() {
    String path = "/Users/mash5/Downloads/3.jpeg";
    
    Mat img = imread(path);
    Mat img_copy = img.clone();
    
    // 灰度图
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    
    // 边缘检测
    Canny(img_gray, img_gray, 50, 200, 3);
    
    // 二值化图像
    Mat img_binary;
    threshold(img_gray, img_binary, 200, 255, THRESH_BINARY);
    imshow("binary", img_binary);
    
    vector<Vec2f> lines;
    /*
     函数原型：
     void HoughLines(InputArray image, OutputArray lines, double rho, double theta, int threshold, double srn=0, double stn=0, double min_theta=0, double max_theta=CV_PI)
     参数详解：
     image，输入图像，即源图像，需为8位的单通道二进制图像。
     lines，经过调用HoughLines函数后储存了霍夫线变换检测到线条的输出矢量。
     每一条线由具有两个元素的矢量表示，其中，是离坐标原点((0,0)（也就是图像的左上角）的距离。 是弧度线条旋转角度（0~垂直线，π/2~水平线）。
     rho，以像素为单位的距离精度。另一种形容方式是直线搜索时的进步尺寸的单位半径。PS:Latex中/rho就表示。
     theta，以弧度为单位的角度精度。另一种形容方式是直线搜索时的进步尺寸的单位角度。
     threshold，累加平面的阈值参数，即识别某部分为图中的一条直线时它在累加平面中必须达到的值。大于阈值threshold的线段才可以被检测通过并返回到结果中。
     srn，有默认值0。对于多尺度的霍夫变换，这是第三个参数进步尺寸rho的除数距离。粗略的累加器进步尺寸直接是第三个参数rho，而精确的累加器进步尺寸为rho/srn。
     stn，有默认值0，对于多尺度霍夫变换，srn表示第四个参数进步尺寸的单位角度theta的除数距离。且如果srn和stn同时为0，就表示使用经典的霍夫变换。
     否则，这两个参数应该都为正数。
     min_theta，对于标准和多尺度Hough变换，检查线条的最小角度。必须介于0和max_theta之间。
     max_theta, 对于标准和多尺度Hough变换，检查线条的最大角度。必须介于min_theta和CV_PI之间.
     */
    HoughLines(img_binary, lines, 1, CV_PI / 180, 100, 0, 0);
    
    for (int i = 0; i < lines.size(); i++) {
        // 极径
        float rho = lines[i][0];
        // 极角
        float theta = lines[i][1];
        printf("rho %.2f, theta : %.2f\n", rho, theta);
        
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        line(img, pt1, pt2, Scalar(128, 128, 0), 1, LINE_AA);
        
    }
    imshow("HoughLines", img);
    
    
    vector<Vec4i> lines_p;
    /*
     函数原型：
     void HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0)
     参数详解：
     image，输入图像，即源图像，需为8位的单通道二进制图像。
     lines，经过调用HoughLinesP函数后后存储了检测到的线条的输出矢量，每一条线由具有四个元素的矢量(x_1,y_1, x_2, y_2）  表示，其中，(x_1, y_1)和(x_2, y_2) 是是每个检测到的线段的结束点。
     rho，以像素为单位的距离精度。另一种形容方式是直线搜索时的进步尺寸的单位半径。
     theta，以弧度为单位的角度精度。另一种形容方式是直线搜索时的进步尺寸的单位角度。
     threshold，累加平面的阈值参数，即识别某部分为图中的一条直线时它在累加平面中必须达到的值。大于阈值threshold的线段才可以被检测通过并返回到结果中。
     minLineLength，有默认值0，表示最低线段的长度，比这个设定参数短的线段就不能被显现出来。
     maxLineGap，有默认值0，允许将同一行点与点之间连接起来的最大的距离。
     */
    HoughLinesP(img_binary, lines_p, 1, CV_PI / 180, 100, 0, 10);
    for (int i = 0; i < lines_p.size(); i++) {
        Vec4i l = lines_p[i];
        line(img_copy, Point(l[0],l[1]), Point(l[2],l[3]), Scalar(128, 128, 0), 1, LINE_AA);
    }
    imshow("HoughLinesP", img_copy);
    
    waitKey(0);
}

static void scharrDemo() {
    String path = "/Users/mash5/Downloads/1.jpeg";
    
    Mat img = imread(path);
    
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    // 先滤波
    GaussianBlur(img_gray, img_gray, Size(3,3), 0);
    imshow("Image Gray", img_gray);
    
    Mat dst_x, abs_dst_x;
    Scharr(img_gray, dst_x, CV_16S, 1, 0);
    convertScaleAbs(dst_x, abs_dst_x);
    
    Mat dst_y, abs_dst_y;
    Scharr(img_gray, dst_y, CV_16S, 0, 1);
    convertScaleAbs(dst_y, abs_dst_y);
    
    Mat dst;
    addWeighted(abs_dst_x, 0.5, abs_dst_y, 0.5, 0, dst);
    
    imshow("Scharr", dst);
    
    waitKey(0);
}

// 拉普拉斯
static void laplacianDemo() {
    String path = "/Users/mash5/Downloads/1.jpeg";
    
    Mat img = imread(path);
    
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    // 先滤波
    GaussianBlur(img_gray, img_gray, Size(3,3), 0);
//    // 提高对比度
//    convertScaleAbs(img_gray, img_gray, 2, -100);
    imshow("Image Gray", img_gray);
    
    Mat dst;
    /*
     Laplacian二阶导数
     函数原型：
     void Laplacian( InputArray src, OutputArray dst, int ddepth, int ksize = 1, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);
     参数详解：
     src，源图像，单通道8位图像。
     dst，输出图像，和原图像一样的类型和尺寸。
     ddepth，输出图像的深度。
     ksize，用于计算二阶导数的滤波器的孔径尺寸，大小必须为正奇数，且有默认值1.
     scale，比例因子，有默认值0。
     delta，结果存入目标图像之前可选的delta值, 有默认值0
     */
    Laplacian(img_gray, dst, CV_16S, 3, 1, 0);
    
    // 转化为8位图像
    convertScaleAbs(dst, dst);
    
//    // 开运算，去白点。
//    Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(-1, -1));
//    morphologyEx(dst, dst, MORPH_OPEN, element);
    
    imshow("laplacian", dst);
    
//    // 原图减拉普拉斯
//    subtract(img_gray, dst, dst);
//    imshow("gray - laplacian", dst);
    
    waitKey(0);
    
}

// 边缘检测
static void cannyDemo() {
    String path = "/Users/mash5/Downloads/1.jpeg";
    
    Mat img = imread(path);
    
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    // 先滤波
    GaussianBlur(img_gray, img_gray, Size(3,3), 0);
    imshow("Image Gray", img_gray);
    
    Mat dstX;
    Mat kernelX = (Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    filter2D(img_gray, dstX, CV_16S, kernelX);
    
    Mat dstY;
    Mat kernelY = (Mat_<char>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    filter2D(img_gray, dstY, CV_16S, kernelY);
    
    int width = dstX.cols;
    int height = dstX.rows;
    Mat dstXY = Mat(height, width, dstX.type());
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int gX = dstX.at<short>(row, col);
            int gY = dstY.at<short>(row, col);
            dstXY.at<short>(row, col) = abs(gX) + abs(gY);//sqrt(pow(gX, 2) + pow(gY, 2));
        }
    }
    /*
     convertScaleAbs函数是OpenCV中的函数，使用线性变换转换输入数组元素成8位无符号整型。
     函数原型：
     void convertScaleAbs(InputArray src, OutputArray dst, double alpha = 1, double beta = 0);
     参数含义：
     src，源图像
     dst，输出图像(深度为 8u).
     scale，乘数因子.
     shift，原图像元素按比例缩放后添加的值。
     用于实现对整个图像数组中的每一个元素，进行如下操作：
     dst(I) = abs(src(I)*scale + shift)
     */
    convertScaleAbs(dstXY, dstXY);
//    threshold(dstXY, dstXY, 50, 255, THRESH_TOZERO);
//    threshold(dstXY, dstXY, 120, 255, THRESH_TRUNC);
    imshow("filter2D xy diff", dstXY);
    
    /*
     Sobel函数使用扩展的Sobel算子，来计算一阶、二阶、三阶或混合图像的差分
     函数原型：
     void Sobel( InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT );
     参数详解：
     src，源图像。
     dst，输出图像。
     ddepth，输出图像的深度。
     dx，x方向的差分阶数。
     dy，y方向的差分阶数。
     ksize，Sobel核的大小，默认3，必须取1、3、5、7.
     */
    Sobel(img_gray, dstX, CV_16S, 1, 0, 3);
    convertScaleAbs(dstX, dstX);
    
    Sobel(img_gray, dstY, CV_16S, 0, 1, 3);
    convertScaleAbs(dstY, dstY);
    
    addWeighted(dstX, 0.5, dstY, 0.5, 0, dstXY);
    imshow("Sobel xy diff", dstXY);
    
    
    Mat dst;
    /*
     函数原型：
     void Canny( InputArray image, OutputArray edges, double threshold1, double threshold2, int apertureSize = 3, bool L2gradient = false);
     参数详解：
     image，源图像，需要为单通道8位图像。
     edges，输出图像，和原图像一样的类型和尺寸。
     threshold1，第一个滞后性阈值。
     threshold2，第二个滞后性阈值。
     apertureSize，表示应用Sobel算子的孔径大小，默认为3.
     需要注意的是，阈值1和阈值2两者中较小的值用于边缘连接，较大的值用来控制强边缘的初始段，推荐高低阈值比在2:1到3:1之间。
     (1) 如果值大于maxVal，则处理为边界
     (2) 如果值minVal<梯度值<maxVal，再检查是否挨着其他边界点，如果旁边没有边界点，则丢弃，如果连着确定的边界点，则也认为其为边界点。
     (3) 梯度值<minVal，舍弃。
     */
    Canny(img_gray, dst, 50, 120, 3);
    imshow("Canny", dst);
    
    waitKey(0);
}

// 高斯差
static void gaussianDiff() {
    String path = "/Users/mash5/Downloads/1.jpeg";
    
    Mat img = imread(path);
    imshow("Image", img);
    
    Mat img_gray, g1, g2, dst;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    
    GaussianBlur(img_gray, g1, Size(5, 5), 0, 0);
    GaussianBlur(g1, g2, Size(5, 5), 0, 0);
    // 高斯差
    subtract(g1, g2, dst);
    // 归一化数据
    normalize(dst, dst, 255, 0, NORM_MINMAX);
    
    imshow("GaussianDiff", dst);
    waitKey(0);
    
}

// 降低一倍采样
static Mat zoomout(Mat &src) {
//    先高斯滤波，再删除奇/偶数行列
//    Mat temp;
//    GaussianBlur(src, temp, Size(5,5), 0);
//    Mat dst = Mat::zeros(temp.rows / 2, temp.cols / 2, temp.type());
//
//    int width = temp.cols;
//    int height = temp.rows;
//    int nc = temp.channels();
//    for (int row = 0; row < height; row++) {
//        if (!(row&1)) {
//            for (int col = 0; col < width; col++) {
//                if (!(col&1)) {
//                    if (nc == 1) {
//                        int g = temp.at<uchar>(row, col);
//                        dst.at<uchar>(row/2, col/2) = g;
//                    } else if (nc == 3) {
//                        int b = temp.at<Vec3b>(row, col)[0];
//                        int g = temp.at<Vec3b>(row, col)[1];
//                        int r = temp.at<Vec3b>(row, col)[2];
//                        dst.at<Vec3b>(row/2, col/2)[0] = b;
//                        dst.at<Vec3b>(row/2, col/2)[1] = g;
//                        dst.at<Vec3b>(row/2, col/2)[2] = r;
//                    }
//                }
//            }
//        }
//    }
    
    Mat dst;
    pyrDown(src, dst);
    
    return dst;
}

// 阈值操作（二值图像）
static void thresholdDemo(){
    String path = "/Users/mash5/Downloads/2.jpeg";
    
    Mat img = imread(path);
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    imshow("Image Gray", img_gray);
    
    Mat dst;
    /*
     阈值操作
     函数原型：
     double threshold(InputArray src, OutputArray dst, double thresh, double maxval, int type);
     参数详解：
     src，源图像。
     dst，输出图像，与源图像大小一致。
     thresh，阈值的具体值。
     maxval，当阈值类型type取THRESH_BINARY或THRESH_BINARY_INV阈值类型时的最大值。
     type，阈值类型，有以下几种类型：
     THRESH_BINARY      二进制阈值化 -> 大于阈值为1 小于阈值为0
     THRESH_BINARY_INV  反二进制阈值化 -> 大于阈值为0 小于阈值为1
     THRESH_TRUNC       截断阈值化 -> 大于阈值为阈值，小于阈值不变
     THRESH_TOZERO      阈值化为0 -> 大于阈值的不变，小于阈值的全为0
     THRESH_TOZERO_INV  反阈值化为0 -> 大于阈值为0，小于阈值不变
     */
    threshold(img_gray, dst, 120, 255, THRESH_BINARY);
    imshow("threshold", dst);
    
    /*
     自适应阈值操作
     函数原型：
     void adaptiveThreshold( InputArray src, OutputArray dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C );
     参数详解：
     src，源图像。
     dst，输出图像，与源图像大小一致。
     maxValue，当阈值类型thresholdType取THRESH_BINARY或THRESH_BINARY_INV阈值类型时的最大值。
     adaptiveMethod，在一个邻域内计算阈值所采用的算法，平均法和高斯法。有两个取值，分别为：
     ADAPTIVE_THRESH_MEAN_C的计算方法是计算出邻域的平均值再减去第七个参数double C的值
     ADAPTIVE_THRESH_GAUSSIAN_C的计算方法是计算出邻域的高斯均值再减去第七个参数double C的值
     
     thresholdType，这是阈值类型，只有两个取值，分别为THRESH_BINARY和THRESH_BINARY_INV
     blockSize：adaptiveThreshold的计算单位是像素的邻域块，这个值作决定了邻域块取多大。
     C，这个参数实际上是一个偏移值调整量
     */
    adaptiveThreshold(img_gray, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);
    imshow("adaptiveThreshold", dst);
    
    waitKey(0);
}

// 图像线性混合
static void linearBlend() {
    String path1 = "/Users/mash5/Downloads/1.jpeg";
    String path2 = "/Users/mash5/Downloads/2.jpeg";
    
    Mat img1 = imread(path1);
    Mat img2 = imread(path2);
    Mat img3 = img1(Rect(10,10, img2.cols, img2.rows));
    Mat img4;
    /*
     void addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype = -1)
     src1，表示需要加权的第一个数组，常常填一个Mat
     alpha，表示第一个数组的权重
     src2，表示第二个数组，需要和第一个数组拥有相同的尺寸和通道数
     beta，第二个数组的权重值，值为1-alpha
     gamma，一个加到权重总和上的标量值。
     dst，输出的数组，和输入的两个数组拥有相同的尺寸和通道数dst = src1[I] * alpha + src2[I] * beta + gamma
     dtype，输出阵列的可选深度，有默认值-1。
     当两个输入数组具有相同深度时，这个参数设置为-1（默认值），即等同于src1.depth()。
     */
    addWeighted(img2, 0.5, img3, 0.5, 0, img4);
    
    imshow("Image addWeighted", img4);
    waitKey(0);
}

// 图像的ROI区域选择与复制
static void roiCopy() {
    String path1 = "/Users/mash5/Downloads/1.jpeg";
    String path2 = "/Users/mash5/Downloads/2.jpeg";
    
    Mat img1 = imread(path1);
    Mat img2 = imread(path2);
    
    /*
     Rect(x,y,width,height)//矩形框
     Rect是一个矩形框。
     x为起始列；
     y为起始行；
     width为宽度；
     height为高度；
     
     Range(start,end)//感兴趣行列范围
     Range是感兴趣起始行/列与终点行/列。
     分别用上面两种方法表示图像img从（100,100）到（200,200）的区域为：
     img(Rect(100, 100, 100, 100));
     img(Range(100, 200), Range(100,200));
     */
    Mat img3 = img1(Rect(10,10, img2.cols, img2.rows));
    imshow("Image ROI", img3);
    
    // 复制img2到ROI区域
    img2.copyTo(img3, img2);
    imshow("Image Copy", img1);
    
    waitKey(0);
}

// 灰度图
static void grayDemo() {
    String path = "/Users/mash5/Downloads/1.jpeg";
    
    Mat img = imread(path);
    imshow("Image", img);
    
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    imshow("Image Gray", img_gray);
    
    waitKey(0);
}

// 绘图
static void drawDemo() {
    String path = "/Users/mash5/Downloads/2.jpeg";
    Mat img = imread(path);
    
    /*
     绘制线
     void line(CV_IN_OUT Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineT ype=8, int shift=0);
     img，要绘制线段的图像。
     pt1，线段的起点。
     pt2，线段的终点。
     color，线段的颜色，通过一个Scalar对象定义。
     thickness，线条的宽度。
     lineType，线段的类型。可以取值8， 4， 和CV_AA， 分别代表8邻接连接线，4邻接连接线和反锯齿连接线。默认值为8邻接。为了获得更好地效果可以选用CV_AA(采用了高斯滤波)。
     shift，坐标点小数点位数。
     */
    line(img, Point(1,1), Point(80,300), Scalar(0, 255, 0));
    
    /*
     绘制矩形
     void rectangle(CV_IN_OUT Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0);
     pt1，矩形的左上角坐标。
     pt2，矩阵的右下角坐标。
     color，线段的颜色，通过一个Scalar对象定义。
     thickness，线条的宽度。
     lineType，线段的类型。可以取值8， 4， 和CV_AA， 分别代表8邻接连接线，4邻接连接线和反锯齿连接线。默认值为8邻接。为了获得更好地效果可以选用CV_AA(采用了高斯滤波)。
     shift，坐标点小数点位数。
     */
    rectangle(img, Point(300,100), Point(400, 500), Scalar(255, 0, 0));
    
    /*
     绘制椭圆
     void ellipse(CV_IN_OUT Mat& img, Point center, Size axes,double angle, double startAngle, double endAngle, const Scalar& color, int thickness=1,int lineType=8, int shift=0);
     img，要绘制椭圆的图像。
     center，椭圆中心点坐标。
     axes，椭圆位于该Size决定的矩形内。（即定义长轴和短轴）。
     angle，椭圆旋转角度。
     startAngle，椭圆开始绘制时角度。
     endAngle，椭圆绘制结束时角度。（若绘制一个完整的椭圆，则startAngle=0, endAngle = 360）。
     color，椭圆的颜色。
     thickness，绘制椭圆线粗。负数表示全部填充。
     lineType，线段的类型。可以取值8， 4， 和CV_AA， 分别代表8邻接连接线，4邻接连接线和反锯齿连接线。默认值为8邻接。为了获得更好地效果可以选用CV_AA(采用了高斯滤波)。
     shift，坐标点小数点位数。
     */
    ellipse(img, Point(100, 400), Size(100, 80), 0, 0, 360, Scalar(0,0,255));
    
    /*
     绘制圆形
     void circle(CV_IN_OUT Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0);
     center，圆心坐标。
     radius，半径。
     color，线段的颜色，通过一个Scalar对象定义。
     thickness，线条的宽度。负数表示全部填充。
     lineType，线段的类型。可以取值8， 4， 和CV_AA， 分别代表8邻接连接线，4邻接连接线和反锯齿连接线。默认值为8邻接。为了获得更好地效果可以选用CV_AA(采用了高斯滤波)。
     shift，坐标点小数点位数。
     */
    circle(img, Point(100, 400), 50, Scalar(30, 255, 0));
    
    /*
     绘制多边形
     void fillPoly(Mat& img, const Point** pts,const int* npts, int ncontours, const Scalar& color, int lineType=8, int shift=0, Point offset=Point());
     pts，多边形定点集。
     npts，多边形的顶点数目。
     ncontours，要绘制多边形的数量。
     color，线段的颜色，通过一个Scalar对象定义。
     thickness，线条的宽度。
     lineType，线段的类型。可以取值8， 4， 和CV_AA， 分别代表8邻接连接线，4邻接连接线和反锯齿连接线。默认值为8邻接。为了获得更好地效果可以选用CV_AA(采用了高斯滤波)。
     shift，坐标点小数点位数。
     offset，所有点轮廓的可选偏移。
     */
    Point root_points[1][5];
    root_points[0][0] = Point(10,100);
    root_points[0][1] = Point(300,20);
    root_points[0][2] = Point(200,300);
    root_points[0][3] = Point(500,100);
    const Point* ppt[1] = {root_points[0]};
    int npt[] = {4};
    fillPoly(img, ppt, npt, 1, Scalar(0, 0, 255));
    
    /*
     绘制文字
     void putText(Mat& img, const string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false)
     img，显示文字所在图像.
     text，待显示的文字.
     org，文字在图像中的左下角 坐标.
     font，字体结构体.
     fontFace，字体类型， 可选择字体：
     FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN,
     FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX,
     FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL,
     FONT_HERSHEY_SCRIPT_SIMPLEX, or FONT_HERSHEY_SCRIPT_COMPLEX,
     以上所有类型都可以配合FONT_HERSHEY_ITALIC使用，产生斜体效果。
     fontScale，字体大小，该值和字体内置大小相乘得到字体大小
     color，文本颜色
     thickness，写字的线的粗细
     lineType，线型.
     bottomLeftOrigin，true，图像数据原点在左下角。否则，图像数据原点在左上角.
     */
    putText(img, "Hello World!", Point(10, 600), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 255, 0));
    
    imshow("Image Draw", img);
    waitKey(0);
}

// 图像元素操作
static void pixelOperation() {
    String path = "/Users/mash5/Downloads/2.jpeg";
    Mat img = imread(path);
    
    // 图像像素位非操作
//    int width = img.cols;
//    int height = img.rows;
//    int nc = img.channels();
//    for (int row = 0; row < height; row++) {
//        for (int col = 0; col < width; col++) {
//            if (nc == 1) {
//                int temp = img.at<uchar>(row, col);
//                img.at<uchar>(row, col) = 255 - temp;
//            } else if (nc == 3) {
//                int b = img.at<Vec3b>(row, col)[0];
//                int g = img.at<Vec3b>(row, col)[1];
//                int r = img.at<Vec3b>(row, col)[2];
//                img.at<Vec3b>(row, col)[0] = 255 - b;
//                img.at<Vec3b>(row, col)[1] = 255 - g;
//                img.at<Vec3b>(row, col)[2] = 255 - r;
//            }
//        }
//    }
//    Mat dst;
//    bitwise_not(img, dst);
//    imshow("Image Result", dst);
    
    
    //  增强对比度
    int width = img.cols;
    int height = img.rows;
    int nc = img.channels();
    float alpha = 1.5;
    float beta = 10;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (nc == 1) {
                float gray = img.at<uchar>(row, col);
                img.at<uchar>(row, col) = saturate_cast<uchar>(gray*alpha + beta);
            } else if (nc == 3) {
                float b = img.at<Vec3b>(row, col)[0];
                float g = img.at<Vec3b>(row, col)[1];
                float r = img.at<Vec3b>(row, col)[2];
                img.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b*alpha + beta);
                img.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g*alpha + beta);
                img.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r*alpha + beta);
            }
        }
    }
    
    imshow("Image Result", img);
    waitKey(0);
}

// 滤波
static void blurDemo() {
    String path = "/Users/mash5/Downloads/2.jpeg";
    
    Mat img = imread(path);
    Mat dst;
    imshow("Image", img);
    
    /*
     均值滤波
     函数原型：
     void blur(InputArray src, OutputArray dst, Size ksize, Point anchor=Point(-1,-1), int borderType=BORDER_DEFAULT)
     参数详解：
     src，输入图像，即源图像，填Mat类的对象即可。
     dst，即目标图像，需要和源图片有一样的尺寸和类型。
     ksize，核的大小。
     anchor，表示锚点（即被平滑的那个点），注意他有默认值Point(-1,-1)。如果这个点坐标是负值的话，就表示取核的中心为锚点，
     所以默认值Point(-1,-1)表示这个锚点在核的中心。
     borderType，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_DEFAULT。
     */
    blur(img, dst, Size(5,5));
    imshow("blur", dst);
    
    /*
     高斯滤波
     函数原型:
     void GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, int borderType=BORDER_DEFAULT);
     参数详解：
     src，输入图像，即源图像，填Mat类的对象即可。
     dst，即目标图像，需要和源图片有一样的尺寸和类型。
     ksize，高斯内核的大小。
     sigmaX，表示高斯核函数在X方向的的标准偏差。
     sigmaY，表示高斯核函数在Y方向的的标准偏差。
     borderType，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_DEFAULT。
     */
    GaussianBlur(img, dst, Size(3, 3), 0, 0);
    imshow("GaussianBlur", dst);
    
    /*
     中值滤波(对椒盐噪音效果好)
     函数原型：void medianBlur(inputArray src,OutputArray dst,int ksize)
     参数详解：
     src，输入图像，即源图像，填Mat类的对象即可。
     dst，即目标图像，需要和源图片有一样的尺寸和类型。
     ksize，孔径的尺寸，参数必须是大于1的奇
     */
    medianBlur(img, dst, 5);
    imshow("medianBlur", dst);
    
    /*
     双边滤波
     函数原型：
     void bilateralFilter( InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType = BORDER_DEFAULT);
     参数详解：
     src，输入图像，即源图像，填Mat类的对象即可。
     dst，即目标图像，需要和源图片有一样的尺寸和类型。
     d, 表示在过滤过程中每个像素邻域的直径。如果这个值我们设其为非正数，那么会从sigmaSpace来计算出它来。
     sigmaColor, 颜色空间滤波器的sigma值。这个参数的值越大，就表明该像素邻域内有更宽广的颜色会被混合到一起，产生较大的半相等颜色区域。
     sigmaSpace, 坐标空间的标注方差。他的数值越大，意味着越远的像素会相互影响，从而使更大的区域足够相似的颜色获取相同的颜色。当d>0，d指定了邻域大小且与sigmaSpace无关。否则，d正比于sigmaSpace。
     borderType，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_DEFAULT。
     */
    bilateralFilter(img, dst, 5, 75, 75);
    imshow("bilateralFilter", dst);
    
    waitKey(0);
}

// 腐蚀
static void erodeDemo() {
    String path = "/Users/mash5/Downloads/2.jpeg";
    
    Mat img = imread(path);
    imshow("Image", img);
    
    Mat dst;
    int cols = (img.cols - 1) * img.channels();
    int offsetx = img.channels();
    int rows = img.rows;
    dst = Mat::zeros(img.size(), img.type());
    for (int row = 1; row < (rows - 1); row++) {
        const uchar * previous = img.ptr<uchar>(row -1);
        const uchar * current = img.ptr<uchar>(row);
        const uchar * next = img.ptr<uchar>(row + 1);
        uchar * output = dst.ptr<uchar>(row);
        for (int col = offsetx; col < cols; col++) {
            uchar min_value = min(previous[col - offsetx], previous[col + offsetx]);
            min_value = min(min_value, previous[col]);
            min_value = min(min_value, current[col - offsetx]);
            min_value = min(min_value, current[col]);
            min_value = min(min_value, current[col + offsetx]);
            min_value = min(min_value, next[col - offsetx]);
            min_value = min(min_value, next[col]);
            min_value = min(min_value, next[col + offsetx]);
            output[col] = min_value;
        }
    }
    imshow("Image Result", dst);
    
    
    Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    Mat dstImg;
    erode(img, dstImg, element);
    imshow("Image Erode", dstImg);
    
    waitKey(0);
}

// 膨胀
static void dilateDemo() {
    String path = "/Users/mash5/Downloads/2.jpeg";
    
    Mat img = imread(path);
    imshow("Image", img);
    
    Mat dst;
    int cols = (img.cols - 1) * img.channels();
    int offsetx = img.channels();
    int rows = img.rows;
    dst = Mat::zeros(img.size(), img.type());
    for (int row = 1; row < (rows - 1); row++) {
        const uchar * previous = img.ptr<uchar>(row -1);
        const uchar * current = img.ptr<uchar>(row);
        const uchar * next = img.ptr<uchar>(row + 1);
        uchar * output = dst.ptr<uchar>(row);
        for (int col = offsetx; col < cols; col++) {
            uchar max_value = max(previous[col - offsetx], previous[col + offsetx]);
            max_value = max(max_value, previous[col]);
            max_value = max(max_value, current[col - offsetx]);
            max_value = max(max_value, current[col]);
            max_value = max(max_value, current[col + offsetx]);
            max_value = max(max_value, next[col - offsetx]);
            max_value = max(max_value, next[col]);
            max_value = max(max_value, next[col + offsetx]);
            output[col] = max_value;
        }
    }
    imshow("Image Result", dst);
    
    
    Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    Mat dstImg;
    dilate(img, dstImg, element);
    imshow("Image Dilate", dstImg);
    
    waitKey(0);
}

// 卷积变换
static void contrastDemo() {
    String path = "/Users/mash5/Downloads/1.jpeg";
    
    Mat img = imread(path);
    imshow("Image", img);
    
    Mat dst;
    int cols = (img.cols - 1) * img.channels();
    int offsetx = img.channels();
    int rows = img.rows;
    dst = Mat::zeros(img.size(), img.type());
    for (int row = 1; row < (rows - 1); row++) {
        const uchar * previous = img.ptr<uchar>(row -1);
        const uchar * current = img.ptr<uchar>(row);
        const uchar * next = img.ptr<uchar>(row + 1);
        uchar * output = dst.ptr<uchar>(row);
        for (int col = offsetx; col < cols; col++) {
            output[col] = saturate_cast<uchar>(5 * current[col] - (current[col - offsetx] + current[col + offsetx] + previous[col] + next[col]));
        }
    }
    imshow("Image Result", dst);
    
    
    Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    Mat dstImg;
    filter2D(img, dstImg, img.depth(), kernel);
    imshow("Image Contrast", dstImg);
    
    waitKey(0);
}
