#ifndef COORDINATE_CONVERTER_H
#define COORDINATE_CONVERTER_H

#include <string>
#include <vector>
#include <math.h>



struct posture // 姿态参数内容结构体
{
    double f;        // 相机焦距、
    double beta;     // 拍摄视野角度
    double R;        // 轮胎半径
    double pixels_n; // 图像像素尺寸
    double pixels_m; // 图像像素尺寸
    double M1;       // 圆心像素值
    double R1;       // 半径像素值
    double theta;    // 圆心偏离角度
};
// L1` = 2 L1 * f * tan(beta) / m
// OM1` = 2 OM1 * f * tan(beta) / m
// theta = arctan((y1-y0)/(x1-x0))



struct camera_intrinsics // 相机内参结构体，通过这四个参数得到3*3的内参矩阵
{
    double fx;
    double fy;
    double u;
    double v;
};

/**
 * @brief 转换各个坐标系之间的坐标
 *
 */
class CoordinateConverter
{
public:
    CoordinateConverter();
    ~CoordinateConverter();

    void getPosition();
    void calculatePosition();
    void calculatePosture();

private:
};
#endif // COORDINATE_CONVERTER_H