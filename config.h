#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// 相机参数结构
struct CameraConfig
{
    cv::Mat K;                     // 内参矩阵
    cv::Mat distCoeffs;            // 畸变系数
    Eigen::Matrix4d extrinsic;     // 外参矩阵
    Eigen::Vector3d camera_offset; // 相机相对于车体中心的偏移 (x, y, z)
};

// 粗定位配置
struct CoarseConfig
{
    CameraConfig cam;             // 车体相机配置
    Eigen::Vector3d laser_offset; // 激光测距模块相对于车体中心的偏移 (x, y, z)
    double max_distance;          // 粗定位最大距离（1000mm）
};

// 精定位配置
struct FineConfig
{
    CameraConfig cam;               // 抓手相机配置
    Eigen::Vector3d gripper_offset; // 抓手相对于相机的偏移 (x, y, z)
    double min_distance;            // 精定位最小距离（300mm）
};